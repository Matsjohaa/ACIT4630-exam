from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# Default URL used by the CLI. This endpoint returns the entire TEK17 regulation
# with guidance (veiledning) in a single HTML "root print" document.
DEFAULT_ROOT_PRINT_URL = (
    "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17"
    "?subtype=root&print=true"
)


@dataclass(frozen=True)
class ManifestRow:
    """
    One manifest entry describing a downloaded snapshot.

    The manifest is append-only JSONL to keep a reproducible log of:
    - what URL was requested
    - what URL was actually served (after redirects)
    - when it was downloaded
    - where the snapshot was stored locally
    - checksum for integrity
    """
    url: str
    final_url: str
    status: int
    downloaded_at: str
    path: str
    sha256: str
    content_type: Optional[str]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _repo_root() -> Path:
    """
    Resolve repository root from this file location:
    src/tek17/corpus/download.py -> repo root is parents[3]
    """
    return Path(__file__).resolve().parents[3]


def _resolve_manifest_path(p: str, repo_root: Path) -> Path:
    """
    Resolve a manifest path robustly.

    Supports:
    - new relative paths stored from repo root, e.g.
      data/raw/dibk_root_print/2026-03-13/tek17_full_root_print.html
    - old absolute paths from the same machine
    - old absolute paths from another machine, if the suffix from /data/... exists
      in the current repo
    """
    path = Path(p)

    if path.is_absolute():
        if path.exists():
            return path

        # Fallback for old absolute paths from another machine:
        # reconstruct from the first 'data' segment if present.
        try:
            parts = path.parts
            if "data" in parts:
                data_idx = parts.index("data")
                rel_from_data = Path(*parts[data_idx:])
                candidate = (repo_root / rel_from_data).resolve()
                return candidate
        except Exception:
            pass

        return path

    return (repo_root / path).resolve()


def _find_latest_snapshot_path(manifest_path: Path, url: str) -> Optional[Path]:
    """
    If the URL has already been downloaded, return the most recent snapshot path
    found in the manifest (last matching row wins).

    Handles both old absolute manifest paths and new repo-relative paths.
    """
    if not manifest_path.exists():
        return None

    repo_root = _repo_root()
    latest: Optional[Path] = None

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        if r.get("url") != url:
            continue

        p = r.get("path") or ""
        if p:
            latest = _resolve_manifest_path(p, repo_root)

    return latest if (latest and latest.exists()) else None


def run_download_root_print(
    url: str,
    out_dir: Path,
    manifest_path: Path,
    force: bool = False,
    timeout_s: int = 60,
    user_agent: str = "tek17-rag-research-bot/0.1 (contact: local-dev)",
) -> Path:
    """
    Download the DiBK TEK17 root-print HTML (full TEK17 with guidance).

    Design intent:
    - We download ONE authoritative HTML snapshot (root print)

    Output:
    - Saves one HTML file under out_dir/<YYYY-MM-DD>/tek17_full_root_print.html
    - Appends one JSONL row to manifest_path
    - Returns the Path to the saved HTML snapshot

    If force=False and the URL already exists in the manifest, we skip downloading
    and return the latest existing snapshot path for that URL.
    """
    repo_root = _repo_root()

    out_dir = (repo_root / out_dir).resolve() if not out_dir.is_absolute() else out_dir.resolve()
    manifest_path = (
        (repo_root / manifest_path).resolve()
        if not manifest_path.is_absolute()
        else manifest_path.resolve()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = _find_latest_snapshot_path(manifest_path, url)
        if existing:
            print(f"URL already in manifest, using existing snapshot: {existing}")
            return existing

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    resp = session.get(url, timeout=timeout_s, allow_redirects=True)
    content_type = resp.headers.get("Content-Type")
    status = resp.status_code
    final_url = resp.url
    body = resp.content or b""
    sha = _sha256_bytes(body) if body else ""

    date_folder = datetime.now().strftime("%Y-%m-%d")
    run_out_dir = out_dir / date_folder
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # Stable filename so the role of this file is obvious when browsing raw data.
    fpath = run_out_dir / "tek17_full_root_print.html"

    if status == 200 and body:
        fpath.write_bytes(body)

    stored_path = ""
    if fpath.exists():
        try:
            stored_path = str(fpath.relative_to(repo_root))
        except ValueError:
            # Fallback if file somehow ends up outside repo root
            stored_path = str(fpath)

    row = ManifestRow(
        url=url,
        final_url=final_url,
        status=status,
        downloaded_at=_now_iso(),
        path=stored_path,
        sha256=sha,
        content_type=content_type,
    )

    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    print(f"Downloaded root-print: status={status} content_type={content_type}")
    print(f"Saved: {fpath}")
    print(f"Manifest: {manifest_path}")

    if not fpath.exists():
        raise RuntimeError(
            f"Download failed or did not produce an HTML snapshot: status={status}"
        )

    return fpath