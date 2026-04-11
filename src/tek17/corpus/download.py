from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_ROOT_PRINT_URL = (
    "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17"
    "?subtype=root&print=true"
)

SNAPSHOT_FILENAME = "tek17_full_root_print.html"
DEFAULT_USER_AGENT = "tek17-rag-research-bot/0.1 (contact: local-dev)"
DEFAULT_ACCEPT_HEADER = "text/html,application/xhtml+xml"


@dataclass(frozen=True)
class ManifestRow:
    """Metadata for one downloaded TEK17 root-print snapshot."""

    url: str
    final_url: str
    status: int
    downloaded_at: str
    path: str
    sha256: str
    content_type: str | None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today_folder_name() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _repo_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    """Resolve a path relative to the repo root unless already absolute."""
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_snapshot_path(path_str: str, repo_root: Path) -> Path:
    """
    Resolve a snapshot path stored in the manifest.

    Supported formats:
    - repo-relative paths
    - absolute paths on the current machine
    - absolute paths from another machine, if the suffix from /data/... exists
      inside the current repository
    """
    path = Path(path_str)

    if not path.is_absolute():
        return (repo_root / path).resolve()

    if path.exists():
        return path

    parts = path.parts
    if "data" in parts:
        data_index = parts.index("data")
        relative_from_data = Path(*parts[data_index:])
        return (repo_root / relative_from_data).resolve()

    return path


def _iter_manifest_rows(manifest_path: Path) -> list[dict]:
    """Read manifest JSONL rows. Malformed lines are skipped."""
    if not manifest_path.exists():
        return []

    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _find_existing_snapshot(manifest_path: Path, url: str, repo_root: Path) -> Path | None:
    """Return the latest existing snapshot path for the given URL, if available."""
    latest_path: Path | None = None

    for row in _iter_manifest_rows(manifest_path):
        if row.get("url") != url:
            continue

        stored_path = row.get("path")
        if not stored_path:
            continue

        candidate = _resolve_snapshot_path(stored_path, repo_root)
        if candidate.exists():
            latest_path = candidate

    return latest_path


def _build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": DEFAULT_ACCEPT_HEADER,
        }
    )
    return session


def _download_snapshot(
    *,
    session: requests.Session,
    url: str,
    timeout_s: int,
) -> tuple[int, str, str | None, bytes]:
    response = session.get(url, timeout=timeout_s, allow_redirects=True)
    return (
        response.status_code,
        response.url,
        response.headers.get("Content-Type"),
        response.content or b"",
    )


def _write_snapshot_file(
    *,
    out_dir: Path,
    status: int,
    body: bytes,
) -> Path:
    run_dir = out_dir / _today_folder_name()
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / SNAPSHOT_FILENAME
    if status == 200 and body:
        snapshot_path.write_bytes(body)

    return snapshot_path


def _to_stored_path(path: Path, repo_root: Path) -> str:
    """Store repo-relative paths when possible, otherwise absolute paths."""
    if not path.exists():
        return ""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _append_manifest_row(manifest_path: Path, row: ManifestRow) -> None:
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def run_download_root_print(
    url: str,
    out_dir: Path,
    manifest_path: Path,
    force: bool = False,
    timeout_s: int = 60,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Path:
    """
    Download the TEK17 root-print HTML and append one entry to the manifest.

    If `force` is false and the URL already exists in the manifest, the latest
    existing snapshot is returned instead of downloading again.
    """
    repo_root = _repo_root()
    resolved_out_dir = _resolve_repo_path(out_dir, repo_root)
    resolved_manifest_path = _resolve_repo_path(manifest_path, repo_root)

    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        existing_snapshot = _find_existing_snapshot(resolved_manifest_path, url, repo_root)
        if existing_snapshot is not None:
            print(f"URL already in manifest, using existing snapshot: {existing_snapshot}")
            return existing_snapshot

    session = _build_session(user_agent)
    status, final_url, content_type, body = _download_snapshot(
        session=session,
        url=url,
        timeout_s=timeout_s,
    )

    snapshot_path = _write_snapshot_file(
        out_dir=resolved_out_dir,
        status=status,
        body=body,
    )

    manifest_row = ManifestRow(
        url=url,
        final_url=final_url,
        status=status,
        downloaded_at=_now_iso(),
        path=_to_stored_path(snapshot_path, repo_root),
        sha256=_sha256_bytes(body) if body else "",
        content_type=content_type,
    )
    _append_manifest_row(resolved_manifest_path, manifest_row)

    print(f"Downloaded root-print: status={status} content_type={content_type}")
    print(f"Saved: {snapshot_path}")
    print(f"Manifest: {resolved_manifest_path}")

    if not snapshot_path.exists():
        raise RuntimeError(
            f"Download failed or did not produce an HTML snapshot: status={status}"
        )

    return snapshot_path