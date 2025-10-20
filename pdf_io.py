# pdf_io.py
import re
from pathlib import Path
from typing import List, Tuple
import requests
from pdfminer.high_level import extract_text
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from bs4 import BeautifulSoup

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

def download_pdf(url: str, dest: Path):
    if not dest.exists():
        r = requests.get(url, timeout=30, headers=UA_HEADERS)
        r.raise_for_status()
        dest.write_bytes(r.content)

def extract_pdf_text(path: Path) -> str:
    txt = extract_text(str(path)) or ""
    txt = txt.replace("\r", "")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def clean_bylaw_text(txt: str) -> str:
    txt = re.sub(r"(?m)^\s*\d+\s*$", "", txt)              # bare page numbers
    txt = re.sub(r"(?m)^\s*\d{2}-\d{3}\s*$", "", txt)      # bylaw numbers
    txt = re.sub(r"(?is)\bContents\b.*?\bCommencement\b", "Commencement", txt)
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)             # join broken lines
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()

def paragraphs_from_text(txt: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
    # drop shouty/all-caps headers and super-short crumbs
    paras = [p for p in paras if len(p) > 60 and not re.fullmatch(r"[A-Z ()\-/\d]{3,}", p)]
    return paras

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

def sentences_from_paragraph(p: str) -> List[str]:
    ensure_nltk()
    return [s.strip() for s in sent_tokenize(p) if s.strip()]

def rank_paragraphs_by_tfidf(question: str, paragraphs: List[str], top_k: int = 6) -> List[int]:
    if not paragraphs:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vec.fit_transform([question] + paragraphs)
    q_vec = X[0]
    doc_mat = X[1:]
    sims = (doc_mat @ q_vec.T).toarray().ravel()
    order = np.argsort(-sims)
    return order[:min(top_k, len(paragraphs))]

# -------- HTML helpers --------

def fetch_html_text(url: str) -> str:
    resp = requests.get(url, timeout=30, headers=UA_HEADERS, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for sel in ["script", "style", "noscript", "nav", "aside", "footer", "header"]:
        for el in soup.select(sel):
            el.decompose()
    parts = []
    for sel in ["h1", "h2", "h3", "h4", "p", "li"]:
        for el in soup.select(sel):
            t = el.get_text(" ", strip=True)
            if t:
                parts.append(t)
    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_text_to_paragraphs(text: str, min_len: int = 60) -> List[str]:
    chunks = []
    for blk in re.split(r"\n{2,}", text):
        blk = blk.strip()
        if len(blk) >= min_len:
            chunks.append(blk)
    return chunks

def build_corpus_with_sources(
    pdf_paragraphs: List[str],
    pdf_source_name: str,
    extra_url_list: List[str]
) -> Tuple[List[str], List[str]]:
    paragraphs: List[str] = []
    sources: List[str] = []

    for p in pdf_paragraphs:
        paragraphs.append(p)
        sources.append(pdf_source_name)

    for url in extra_url_list:
        try:
            html_text = fetch_html_text(url)
            for chunk in chunk_text_to_paragraphs(html_text):
                paragraphs.append(chunk)
                sources.append(url)
        except Exception as e:
            print(f"[warn] failed to fetch {url}: {e}")

    return paragraphs, sources
