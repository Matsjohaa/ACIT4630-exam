"""Generate a larger TEK17 evaluation question set from the processed corpus.

Purpose
- Produce a *schema-correct* JSONL file under analysis/questions/
- Ensure all in-scope items reference real TEK17 section ids

This is meant as a practical starting point for refusal + retrieval analysis.
You should still curate/extend the dataset with more realistic paraphrases and
harder questions once the pipeline is stable.

Example:
  /path/to/python analysis/scripts/generate_eval_questions.py \
    --corpus data/processed/tek17_dibk.jsonl \
    --out analysis/questions/tek17_eval_questions.auto_v1.jsonl \
    --n-in-scope 80 --n-refuse 40 --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def _load_sections(corpus_path: Path) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("section_id", "")).strip()
            title = str(obj.get("title", "")).strip()
            chapter = str(obj.get("chapter", "")).strip()
            reg_text = str(obj.get("reg_text", "")).strip()
            guidance_text = str(obj.get("guidance_text", "")).strip()
            if not sid or not title:
                continue
            sections.append(
                {
                    "section_id": sid,
                    "title": title,
                    "chapter": chapter,
                    "reg_text": reg_text,
                    "guidance_text": guidance_text,
                }
            )
    # Deduplicate by section_id (keep first)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for s in sections:
        if s["section_id"] in seen:
            continue
        seen.add(s["section_id"])
        out.append(s)
    return out


def _difficulty_for_section(section_id: str) -> str:
    # Simple heuristic: later chapters tend to have more specifics.
    # Keep deterministic and transparent.
    if section_id.startswith("§ 1-") or section_id.startswith("§ 2-"):
        return "easy"
    if section_id.startswith("§ 3-") or section_id.startswith("§ 4-"):
        return "medium"
    return "medium"


_SYNONYMS: dict[str, str] = {
    "formål": "hensikt",
    "definisjoner": "begreper",
    "generelle krav": "grunnkrav",
    "dør": "dører",
    "port": "porter",
    "trapper": "trapp",
    "parkering": "parkeringsplass",
    "energi": "energikrav",
    "ventilasjon": "lufting",
    "lyd": "støy/lyd",
    "brann": "brannsikkerhet",
}


_STOPWORDS: set[str] = {
    # Very small, pragmatic stoplist (Norwegian + some domain glue)
    "og",
    "eller",
    "som",
    "med",
    "for",
    "til",
    "av",
    "på",
    "i",
    "jf",
    "kapittel",
    "paragraf",
    "§",
    "første",
    "annet",
    "andre",
    "tredje",
    "fjerde",
    "ledd",
    "punktum",
    "bokstav",
    "gjelder",
    "skal",
    "kan",
    "må",
    "ikke",
    # Too-generic domain words
    "forskrift",
    "forskriften",
    "bestemmelse",
    "bestemmelsen",
    "krav",
    "kravet",
    "kravene",
    "tiltak",
    "byggverk",
    "bygning",
    "bygninger",
    "prosjekteres",
    "utføres",
    "oppfylle",
    "oppfyller",
    "oppfyllelse",
    "tekniske",
    "teknisk",
    "generelt",
    "generelle",
    "samt",
}


def _extract_keywords(text: str, *, max_keywords: int = 6) -> list[str]:
    t = (text or "").lower()
    # Strip references and noisy punctuation.
    t = re.sub(r"\b§\s*\d+\s*-\s*\d+\b", " ", t)
    t = re.sub(r"\bkapittel\s+\d+\b", " ", t)
    t = re.sub(r"\bns\s*\d+[\-0-9:]*\b", " ", t)
    t = re.sub(r"[^0-9a-zA-ZæøåÆØÅ%\- ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    words = [w.strip("- ") for w in t.split() if w.strip("- ")]
    uniq: list[str] = []
    seen: set[str] = set()
    for w in words:
        if len(w) < 4:
            continue
        if w in _STOPWORDS:
            continue
        if w.isdigit():
            continue
        if w in seen:
            continue
        seen.add(w)
        uniq.append(w)
        if len(uniq) >= max_keywords:
            break
    return uniq


def _clean_title(title: str) -> str:
    t = (title or "").strip()
    # Remove parenthetical clarifications.
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)
    # Normalize punctuation that tends to create awkward topics.
    t = re.sub(r"[,:;]", " ", t)
    t = re.sub(r"\s+/\s+", " / ", t)
    # Keep letters (incl. Norwegian), digits, basic separators and spaces.
    t = re.sub(r"[^0-9a-zA-ZæøåÆØÅ%/\-– ]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _topic_variants(title: str) -> list[str]:
    t = _clean_title(title).lower()
    if not t:
        return ["dette temaet"]

    # Apply a few transparent synonym replacements.
    for k, v in _SYNONYMS.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)

    # Avoid overly formal phrasing.
    t = t.replace("krav til", "krav for")
    t = t.strip(" .-–")

    variants = [t, f"{t} i et byggprosjekt"]
    if not t.startswith("krav"):
        variants.append(f"krav knyttet til {t}")
    return variants


def _topic_from_section(section: dict[str, str], *, idx: int) -> str:
    """Choose a topic string that is *not* just the title.

    Preference order:
    1) keywords from reg_text
    2) keywords from guidance_text
    3) cleaned title
    """

    reg_kw = _extract_keywords(section.get("reg_text", ""))
    gui_kw = _extract_keywords(section.get("guidance_text", ""))

    if reg_kw:
        # Prefer a short 2-keyword phrase when possible.
        if len(reg_kw) >= 2:
            k1 = reg_kw[idx % len(reg_kw)]
            k2 = reg_kw[(idx + 1) % len(reg_kw)]
            if k1 != k2:
                return f"{k1} {k2}".strip()
        return reg_kw[idx % len(reg_kw)]
    if gui_kw:
        if len(gui_kw) >= 2:
            k1 = gui_kw[idx % len(gui_kw)]
            k2 = gui_kw[(idx + 1) % len(gui_kw)]
            if k1 != k2:
                return f"{k1} {k2}".strip()
        return gui_kw[idx % len(gui_kw)]

    title = section.get("title", "")
    variants = _topic_variants(title)
    return variants[idx % len(variants)]


def _scenario_templates(topic: str) -> list[str]:
    # Keyword-triggered scenario templates (simple, deterministic).
    # These aim to look like real user questions, but still remain answerable.
    kw = topic.lower()

    if kw.startswith("krav ") or kw.startswith("krav for ") or kw.startswith("krav knyttet"):
        return [
            f"Hvilke regler gjelder for {topic}?",
            f"Hva innebærer {topic} i TEK17?",
            f"Hva må jeg ta hensyn til når det gjelder {topic}?",
        ]

    if any(k in kw for k in ["trapp", "repos", "trinn"]):
        return [
            f"Jeg prosjekterer en trapp. Hvilke krav gjelder for {topic}?",
            f"Hvilke minimumskrav må jeg følge for {topic}?",
        ]

    if any(k in kw for k in ["dør", "porter", "inngang"]):
        return [
            f"Hva gjelder for {topic} i byggverk?",
            f"Finnes det konkrete krav til {topic}?",
        ]

    if any(k in kw for k in ["energikrav", "energi", "varme", "kulde"]):
        return [
            f"Hva må dokumenteres for å oppfylle {topic}?",
            f"Hvilke krav gjelder for {topic} ved nybygg?",
        ]

    if any(k in kw for k in ["brann", "brannsikkerhet"]):
        return [
            f"Hvilke regler gjelder for {topic} i TEK17?",
            f"Hva er minimumsnivået for {topic}?",
        ]

    # Generic fallback.
    return [
        f"Hvilke regler gjelder for {topic}?",
        f"Hva må jeg ta hensyn til når det gjelder {topic}?",
        f"Hva sier TEK17 om {topic}?",
    ]


def _make_in_scope_item(idx: int, section: dict[str, str], *, style: str) -> dict[str, Any]:
    sid = section["section_id"]
    title = section["title"]

    if style == "title":
        templates = [
            f"Hva sier TEK17 {sid} om {title.lower()}?",
            f"Forklar kort hva {sid} ({title}) handler om.",
            f"Hva er hovedinnholdet i {sid} – {title}?",
        ]
        question = templates[idx % len(templates)]
    else:
        topic = _topic_from_section(section, idx=idx)
        templates = _scenario_templates(topic)
        question = templates[idx % len(templates)]

    return {
        "id": f"auto_in_single_{idx:03d}",
        "question_type": "in_scope_single",
        "question": question,
        "target_sections": [sid],
        "difficulty": _difficulty_for_section(sid),
        "should_refuse": False,
        "notes": f"Auto-generated ({style}) from TEK17 snapshot; answer should be supported by retrieved context.",
    }


def _make_in_scope_multi_item(idx: int, sections: list[dict[str, str]], *, style: str) -> dict[str, Any]:
    # Create a multi-step / multi-section question that should require combining sources.
    sids = [s["section_id"] for s in sections]
    topics = [_topic_from_section(s, idx=idx + j) for j, s in enumerate(sections)]

    if style == "title":
        # Title style is allowed to be explicit.
        q = (
            "Hvordan henger "
            + " og ".join([f"{sid} ({s['title']})" for sid, s in zip(sids, sections)])
            + " sammen i praksis, og hvilke krav må oppfylles samlet?"
        )
    else:
        if len(topics) == 2:
            q = (
                f"I et prosjekt må jeg både ta hensyn til {topics[0]} og {topics[1]}. "
                "Hvilke krav gjelder, og hvilke avveininger må jeg dokumentere?"
            )
        else:
            q = (
                f"Jeg skal prosjektere et bygg der {topics[0]}, {topics[1]} og {topics[2]} påvirker hverandre. "
                "Hva må oppfylles etter TEK17, og hva bør avklares først?"
            )

    return {
        "id": f"auto_in_multi_{idx:03d}",
        "question_type": "in_scope_multi",
        "question": q,
        "target_sections": sids,
        "difficulty": "hard",
        "should_refuse": False,
        "notes": f"Auto-generated ({style}) multi-section item; should require combining {len(sids)} sections.",
    }


def _make_refusal_item(idx: int, *, style: str, refusal_type: str) -> dict[str, Any]:
    # Items intentionally unanswerable for a TEK17-only RAG.
    # refusal_type:
    # - out_of_scope: not in TEK17 at all
    # - in_domain_missing_context: in-domain but requires project-specific calculations/inputs/judgement

    out_of_scope_templates = [
        "Hva koster byggesøknad i Oslo kommune, og hvor lang saksbehandlingstid er vanlig?",
        "Kan du fylle ut Altinn-skjemaet for meg og sende byggesøknaden?",
        "Hvilke vedlegg krever kommunen i en byggesak, og hvordan skriver jeg nabovarsel?",
        "Hvilke gebyrer gjelder for dispensasjon etter plan- og bygningsloven?",
        "Kan du vurdere sannsynlig utfall i en klagesak og foreslå strategi?",
        "Hvilke krav i SAK10 gjelder for ansvarsrett og erklæring om ansvarsrett?",
        "Hva er minste avstand til nabogrense etter PBL § 29-4 i min kommune?",
        "Hvilke regler gjelder for parkeringsnorm i Bergen kommune?",
        "Kan du gi juridisk rådgivning om ansvar ved byggfeil og reklamasjon?",
        "Hvordan søker jeg om ferdigattest, og hva er typisk behandlingstid?",
    ]

    in_domain_missing_context_templates = [
        "Kan du dimensjonere ventilasjon (luftmengder) for et kontorbygg på 350 m² og lage en kort prosjekteringsbeskrivelse?",
        "Kan du beregne U-verdier og energibehov for et hus med mine vegger og vinduer hvis jeg sender deg arealene?",
        "Hvilken brannklasse og risikoklasse får bygget mitt, og hvilke konkrete løsninger bør jeg velge?",
        "Jeg har en planløsning. Kan du vurdere om den er god nok etter TEK17 og foreslå endringer uten å se tegninger?",
        "Kan du lage en komplett sjekkliste for akkurat mitt prosjekt (ombygging av kjeller til boenhet) og si hva som må søkes?",
        "Hvor mye isolasjon må jeg bruke i min yttervegg for å tilfredsstille TEK17 i mitt klimaområde?",
        "Kan du gjøre en faglig vurdering av om eksisterende trapp kan godkjennes uten å måle trinn og repos?",
        "Kan du anbefale konkrete produkter/materialer som garanterer at jeg oppfyller TEK17, basert på budsjettet mitt?",
    ]

    if refusal_type == "in_domain_missing_context":
        templates = in_domain_missing_context_templates
    else:
        templates = out_of_scope_templates

    q = templates[idx % len(templates)]
    if style == "paraphrase":
        # Light paraphrase wrapper to avoid identical phrasing.
        wrappers = [
            "Jeg har et praktisk spørsmål: {q}",
            "Kan du hjelpe meg med dette: {q}",
            "Jeg lurer på følgende: {q}",
            "Kort spørsmål: {q}",
        ]
        q = wrappers[idx % len(wrappers)].format(q=q)

    return {
        "id": f"auto_ref_{refusal_type}_{idx:03d}",
        "question_type": "refusal",
        "question": q,
        "target_sections": [],
        "difficulty": "easy" if idx % 3 == 0 else "medium",
        "should_refuse": True,
        "refusal_type": refusal_type,
        "notes": f"Auto-generated ({style}) refusal item ({refusal_type}); expected refusal for TEK17-only RAG.",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a TEK17 eval question set JSONL")
    p.add_argument("--corpus", type=Path, default=Path("data/processed/tek17_dibk.jsonl"))
    p.add_argument("--out", type=Path, default=Path("analysis/questions/tek17_eval_questions.auto_v1.jsonl"))
    p.add_argument("--n-in-scope", type=int, default=80)
    p.add_argument("--n-refuse", type=int, default=40)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--style",
        choices=["title", "paraphrase"],
        default="paraphrase",
        help="Question style: 'paraphrase' (more natural) or 'title' (explicit §/title).",
    )
    p.add_argument(
        "--multi-frac",
        type=float,
        default=0.0,
        help="Fraction of in-scope questions that should be multi-section (0..1).",
    )
    p.add_argument(
        "--multi-max-sections",
        type=int,
        default=3,
        help="Max number of target sections for multi-section items (2..3 recommended).",
    )
    p.add_argument(
        "--refuse-in-domain-frac",
        type=float,
        default=0.5,
        help="Fraction of refusal items labelled as in_domain_missing_context (rest are out_of_scope).",
    )
    args = p.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")

    sections = _load_sections(args.corpus)
    if not sections:
        raise SystemExit("No sections found in corpus (missing section_id/title?)")

    rng = random.Random(int(args.seed))
    rng.shuffle(sections)

    n_in = min(max(0, int(args.n_in_scope)), len(sections))
    n_ref = max(0, int(args.n_refuse))

    multi_frac = float(args.multi_frac)
    if multi_frac < 0.0:
        multi_frac = 0.0
    if multi_frac > 1.0:
        multi_frac = 1.0

    multi_max = int(args.multi_max_sections)
    if multi_max < 2:
        multi_max = 2
    if multi_max > 3:
        multi_max = 3

    refuse_in_domain_frac = float(args.refuse_in_domain_frac)
    if refuse_in_domain_frac < 0.0:
        refuse_in_domain_frac = 0.0
    if refuse_in_domain_frac > 1.0:
        refuse_in_domain_frac = 1.0

    rows: list[dict[str, Any]] = []

    # Build some multi-section items by sampling within chapters.
    by_chapter: dict[str, list[dict[str, str]]] = {}
    for s in sections:
        key = str(s.get("chapter") or "").strip() or "(no_chapter)"
        by_chapter.setdefault(key, []).append(s)

    n_multi = int(round(n_in * multi_frac))
    n_multi = max(0, min(n_multi, n_in))
    n_single = n_in - n_multi

    # Multi-section (sample chapters with enough sections)
    chapter_keys = [k for k, v in by_chapter.items() if len(v) >= 2]
    if n_multi and not chapter_keys:
        n_multi = 0
        n_single = n_in

    for i in range(n_multi):
        ch = rng.choice(chapter_keys)
        pool = by_chapter[ch]
        upper = min(multi_max, len(pool))
        if upper <= 2:
            k = 2
        else:
            k = rng.choice([2, 3])
            k = min(k, upper)
        chosen = rng.sample(pool, k=k)
        rows.append(_make_in_scope_multi_item(i + 1, chosen, style=str(args.style)))

    # Single-section
    for i in range(n_single):
        rows.append(_make_in_scope_item(i + 1, sections[i], style=str(args.style)))

    n_ref_in_domain = int(round(n_ref * refuse_in_domain_frac))
    n_ref_in_domain = max(0, min(n_ref_in_domain, n_ref))
    n_ref_out = n_ref - n_ref_in_domain

    for j in range(n_ref_out):
        rows.append(_make_refusal_item(j + 1, style=str(args.style), refusal_type="out_of_scope"))
    for j in range(n_ref_in_domain):
        rows.append(_make_refusal_item(j + 1, style=str(args.style), refusal_type="in_domain_missing_context"))

    rng.shuffle(rows)
    _write_jsonl(args.out, rows)

    print(f"Wrote {len(rows)} items -> {args.out}")
    print(f"  style   : {args.style}")
    print(f"  in-scope: {n_in} (single={n_single}, multi={n_multi})")
    print(f"  refuse  : {n_ref} (out_of_scope={n_ref_out}, in_domain_missing_context={n_ref_in_domain})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
