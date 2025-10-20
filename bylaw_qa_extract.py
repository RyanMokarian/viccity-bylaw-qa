# bylaw_qa_extract.py
# pip install transformers torch requests pdfminer.six nltk scikit-learn regex beautifulsoup4

import json, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import nltk
from nltk.tokenize import PunktSentenceTokenizer

from qa_engine import QASystem, QAConfig
from pdf_io import (
    download_pdf,
    extract_pdf_text,
    clean_bylaw_text,
    paragraphs_from_text,
    rank_paragraphs_by_tfidf,
    build_corpus_with_sources,
)
from questions import QUESTIONS

# -----------------------------
# Sources
# -----------------------------
PDF_URL = "https://www.victoria.ca/media/file/short-term-rental-regulation-bylaw-18-036-0"
RAW_PDF = Path("bylaw_18_036.pdf")

EXTRA_URLS: List[str] = [
    # City of Victoria – principal residence + 160-night cap
    "https://www.victoria.ca/building-business/business-licensing/short-term-rentals",
    # Province of BC – registry + display rules
    "https://www2.gov.bc.ca/gov/content/housing-tenancy/short-term-rentals/registry/host-registration",
    "https://www2.gov.bc.ca/gov/content/housing-tenancy/short-term-rentals/registry/platform-registration/platform-requirements",
    "https://www2.gov.bc.ca/gov/content/housing-tenancy/short-term-rentals/short-term-rental-legislation",
    "https://gov.bc.ca/ShortTermRental",
]

OUT_JSON = Path("indicators_ml.json")

# -----------------------------
# Label-specific retrieval & validation
# -----------------------------

# Keyword hints to pre-filter candidate paragraphs (increases precision)
LABEL_KEYWORDS: Dict[str, List[str]] = {
    "requires_license": [
        "licence", "license", "business licence", "business license", "must not operate", "operate a short-term rental"
    ],
    "principal_residence_only": [
        "principal residence", "principal dwelling", "primary residence"
    ],
    "display_license_on_listing": [
        "licence number", "license number", "included in", "advertising", "listing", "ad", "include the licence number"
    ],
    "provincial_registration_required": [
        "provincial", "province", "register", "registry", "registration number",
        "must register", "must be registered", "display", "must display", "short-term rental registry"
    ],
    "max_entire_home_nights": [
        "night", "nights", "per year", "per calendar year",
        "entire home", "entire unit", "entire dwelling", "while you are away"
    ],
}

# Preferred source domains/URLs per label (tie-breaks & fallback)
LABEL_PREFERRED_SOURCES: Dict[str, List[str]] = {
    # Municipal requirements live in the bylaw and the City information page
    "requires_license": [
        "victoria.ca/media/file/short-term-rental-regulation-bylaw-18-036-0",
        "victoria.ca/building-business/business-licensing/short-term-rentals",
    ],
    "principal_residence_only": [
        "victoria.ca/building-business/business-licensing/short-term-rentals",
        "victoria.ca/media/file/short-term-rental-regulation-bylaw-18-036-0",
    ],
    "display_license_on_listing": [
        "victoria.ca/media/file/short-term-rental-regulation-bylaw-18-036-0",
    ],
    # Provincial requirements live on gov.bc.ca and www2.gov.bc.ca
    "provincial_registration_required": [
        "www2.gov.bc.ca/gov/content/housing-tenancy/short-term-rentals/registry/host-registration",
        "www2.gov.bc.ca/gov/content/housing-tenancy/short-term-rentals/registry/platform-registration/platform-requirements",
        "gov.bc.ca/ShortTermRental",
        "www2.gov.bc.ca/gov/content/housing-tenancy/short-term-rentals/short-term-rental-legislation",
    ],
    "max_entire_home_nights": [
        "victoria.ca/building-business/business-licensing/short-term-rentals",
    ],
}

# Evidence must match these regexes (strong signal the answer is correct)
LABEL_VALIDATORS: Dict[str, re.Pattern] = {
    # e.g., “must/shall … licence” and “advertising/operate”
    "requires_license": re.compile(r"\b(licen[sc]e)\b.*\b(advertis|operat)", re.I),
    # “principal residence / principal dwelling”
    "principal_residence_only": re.compile(r"\bprincipal\s+(residence|dwelling)\b", re.I),
    # “licence number” AND “advertising|listing|ad”
    "display_license_on_listing": re.compile(r"\blicen[sc]e\s+number\b.*\b(advertis|listing|advertisement|ad)\b", re.I),
    # “must register / registration number / registry”
    "provincial_registration_required": re.compile(
        r"\b(must\s+register|registration\s+number|short-?term\s+rental\s+registry|must\s+be\s+registered)\b", re.I
    ),
    # “<num> nights … per calendar year”
    "max_entire_home_nights": re.compile(r"\b\d{1,3}\s+nights?\b.*?(?:per|in a)\s+(?:calendar\s+)?year", re.I),
}

# -----------------------------
# Helpers
# -----------------------------

def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

def _clean_spaces(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def evidence_snippet_from_span(paragraph: str, start_char: int, end_char: int, window: int = 0) -> str:
    """
    Return a concise evidence snippet: the sentence containing [start_char:end_char],
    optionally including +/- `window` neighboring sentences. Newlines removed.
    """
    if start_char is None or end_char is None or start_char < 0 or end_char > len(paragraph):
        trimmed = paragraph if len(paragraph) <= 500 else paragraph[:500] + "…"
        return _clean_spaces(trimmed)

    _ensure_punkt()
    tok = PunktSentenceTokenizer()
    spans = list(tok.span_tokenize(paragraph))
    mid = (start_char + end_char) // 2

    sent_idx = None
    for i, (s, e) in enumerate(spans):
        if s <= mid < e:
            sent_idx = i
            break
    if sent_idx is None:
        trimmed = paragraph if len(paragraph) <= 500 else paragraph[:500] + "…"
        return _clean_spaces(trimmed)

    i0 = max(0, sent_idx - window)
    i1 = min(len(spans) - 1, sent_idx + window)
    s_char = spans[i0][0]
    e_char = spans[i1][1]
    snippet = paragraph[s_char:e_char]
    snippet = snippet if len(snippet) <= 500 else snippet[:500] + "…"
    return _clean_spaces(snippet)

def keyword_candidate_indices(label_key: str, paragraphs: List[str]) -> List[int]:
    hints = LABEL_KEYWORDS.get(label_key, [])
    if not hints:
        return []
    key_re = re.compile("|".join(re.escape(h) for h in hints), re.I)
    return [i for i, p in enumerate(paragraphs) if key_re.search(p)]

def _source_score(label_key: str, src: str) -> int:
    prefs = LABEL_PREFERRED_SOURCES.get(label_key, [])
    # higher score if source URL contains any preferred fragment
    for rank, frag in enumerate(prefs[::-1], start=1):
        if frag in src:
            return rank  # later items get higher rank slightly
    return 0

def _rank_idxs(label_key: str, question: str, paragraphs: List[str], sources: List[str], pool: List[int], k: int) -> List[int]:
    """TF-IDF rank a pool, then stable sort by source preference for the label."""
    pool_paras = [paragraphs[i] for i in pool]
    order_in_pool = rank_paragraphs_by_tfidf(question, pool_paras, top_k=max(k*2, k))
    ranked = [pool[i] for i in order_in_pool]
    # Stable sort by source preference (desc)
    ranked.sort(key=lambda i: _source_score(label_key, sources[i]), reverse=True)
    return ranked[:k]

def _validate_label(label_key: str, evidence: str, source: str) -> bool:
    patt = LABEL_VALIDATORS.get(label_key)
    if not patt:
        return True
    if not evidence:
        return False
    if not patt.search(evidence):
        return False
    # Optional: for provincial label, insist on provincial domain
    if label_key == "provincial_registration_required":
        if not ("gov.bc.ca" in source or "www2.gov.bc.ca" in source):
            return False
    return True

def select_top_idxs(label_key: str, question: str, paragraphs: List[str], sources: List[str], top_k: int) -> List[int]:
    """Hybrid retrieval with keyword pool + TF-IDF + source preference."""
    cand_idxs = keyword_candidate_indices(label_key, paragraphs)
    pool = cand_idxs if cand_idxs else list(range(len(paragraphs)))
    return _rank_idxs(label_key, question, paragraphs, sources, pool, top_k)

# -----------------------------
# QA + validation loop per label
# -----------------------------

def best_answer_for_label(
    qa: QASystem,
    cfg: QAConfig,
    paragraphs: List[str],
    paragraph_sources: List[str],
    label_key: str
) -> Dict[str, Any]:
    """
    Try multiple paraphrased questions; retrieve; run QA; validate evidence.
    If the best candidate fails validation, retry on a stricter pool (keywords + preferred sources).
    """
    def _span_to_result(idx: int, span: Dict[str, Any]) -> Dict[str, Any]:
        # concise evidence: sentence containing the span
        snippet = evidence_snippet_from_span(
            paragraphs[idx],
            span.get("start_char"),
            span.get("end_char"),
            window=0
        )
        return {
            "answer": _clean_spaces(span["answer"]) if span.get("answer") else "",
            "score": float(span.get("score", 0.0)),
            "para_idx": idx,
            "evidence": snippet,
            "source": paragraph_sources[idx],
        }

    # pass 1: standard retrieval
    best: Dict[str, Any] = {"answer": "", "score": 0.0, "para_idx": None, "evidence": "", "source": ""}

    q_list = QUESTIONS[label_key]
    for q in q_list:
        top_idxs = select_top_idxs(label_key, q, paragraphs, paragraph_sources, top_k=cfg.top_k_chunks)
        for idx in top_idxs:
            span = qa.answer_span(q, paragraphs[idx], cfg)
            if span["answer"] and span["score"] > best["score"]:
                cand = _span_to_result(idx, span)
                if _validate_label(label_key, cand["evidence"], cand["source"]):
                    best = cand
                else:
                    # keep as contingent best if nothing else passes
                    if cand["score"] > best["score"]:
                        best = cand

    # If validation failed for the top pick, run pass 2 with stricter pool
    if not _validate_label(label_key, best["evidence"], best["source"]):
        strict_pool = keyword_candidate_indices(label_key, paragraphs)

        # If we have source prefs, restrict to those sources
        prefs = LABEL_PREFERRED_SOURCES.get(label_key, [])
        if prefs:
            strict_pool = [i for i in strict_pool if any(p in paragraph_sources[i] for p in prefs)] or strict_pool

        if strict_pool:
            strict_idxs = _rank_idxs(label_key, q_list[0], paragraphs, paragraph_sources, strict_pool, cfg.top_k_chunks)
            strict_best: Dict[str, Any] = {"answer": "", "score": 0.0, "para_idx": None, "evidence": "", "source": ""}
            for idx in strict_idxs:
                # try all question variants again on restricted pool
                for q in q_list:
                    span = qa.answer_span(q, paragraphs[idx], cfg)
                    if span["answer"]:
                        cand = _span_to_result(idx, span)
                        if _validate_label(label_key, cand["evidence"], cand["source"]) and cand["score"] > strict_best.get("score", 0.0):
                            strict_best = cand
            if strict_best["answer"]:
                best = strict_best  # only replace if we found a valid one

    return best

# -----------------------------
# Main
# -----------------------------

def main():
    print("1) Fetching PDF…")
    download_pdf(PDF_URL, RAW_PDF)

    print("2) Extracting text…")
    raw = extract_pdf_text(RAW_PDF)
    cleaned = clean_bylaw_text(raw)
    pdf_paragraphs = paragraphs_from_text(cleaned)
    if not pdf_paragraphs:
        raise RuntimeError("No paragraphs extracted from PDF.")
    print(f"   Parsed {len(pdf_paragraphs)} PDF paragraphs.")

    # Build combined corpus (PDF shows as URL for provenance)
    paragraphs, paragraph_sources = build_corpus_with_sources(
        pdf_paragraphs=pdf_paragraphs,
        pdf_source_name=PDF_URL,         # record the URL (not the filename)
        extra_url_list=EXTRA_URLS
    )
    print(f"   Total paragraphs including webpages: {len(paragraphs)}")
    web_ct = sum(1 for s in paragraph_sources if s.startswith("http"))
    print(f"   Web paragraphs: {web_ct}")

    qa = QASystem("distilbert-base-uncased-distilled-squad")
    cfg = QAConfig(max_seq_len=384, doc_stride=128, top_k_chunks=12)

    print("3) Running QA…")
    results: Dict[str, Any] = {}

    # a) booleans
    for key in ["requires_license", "principal_residence_only", "display_license_on_listing", "provincial_registration_required"]:
        ans = best_answer_for_label(qa, cfg, paragraphs, paragraph_sources, key)
        value = bool(ans["answer"]) and ans["score"] >= 0.12

        # After robust validation above, value reflects correctness more reliably
        results[key] = {
            "value": value,
            "confidence": round(float(ans.get("score", 0.0)), 3),
            "answer_span": ans.get("answer", ""),
            "evidence": ans.get("evidence", ""),
            "source": ans.get("source", ""),
        }

    # b) nights
    key = "max_entire_home_nights"
    ans = best_answer_for_label(qa, cfg, paragraphs, paragraph_sources, key)

    nights = None
    # Extract ONLY if the span mentions night(s)
    if ans["answer"] and re.search(r"\bnights?\b", ans["answer"], flags=re.I):
        m = re.search(r"\b(\d{1,3})\b", ans["answer"])
        if m:
            nights = int(m.group(1))

    # Otherwise, search evidence for "<num> nights ... per calendar year"
    if nights is None and ans["evidence"]:
        m = re.search(
            r"\b(\d{1,3})\s+nights?\b.*?(?:per|in a)\s+(?:calendar\s+)?year",
            ans["evidence"],
            flags=re.I,
        )
        if m:
            nights = int(m.group(1))
            if not ans["answer"]:
                ans["answer"] = f"{nights} nights per calendar year"

    results[key] = {
        "value": nights,
        "confidence": round(float(ans.get("score", 0.0)), 3),
        "answer_span": ans.get("answer", ""),
        "evidence": ans.get("evidence", ""),
        "source": ans.get("source", ""),
    }

    out = {"source_pdf": PDF_URL, **results}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Wrote: {OUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
