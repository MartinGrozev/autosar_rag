# utils_text_processing.py

import re
import os
import fitz  # PyMuPDF
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# — Stopwords setup (as before) —
try:
    from nltk.corpus import stopwords
    STOP_WORDS_NLTK = list(stopwords.words('english'))
except ImportError:
    print("⚠️ NLTK not installed; using basic stopword list.")
    STOP_WORDS_NLTK = [
        'all','just','being','over','both','through','yourselves','its','before','herself',
        'had','should','to','only','under','ours','has','do','them','his','very','they',
        'not','during','now','him','nor','did','this','she','each','further','where','few',
        'because','doing','some','are','our','ourselves','out','what','for','while','does',
        'above','between','t','be','we','who','were','here','hers','by','on','about','of',
        'against','s','or','own','into','yourself','down','mightn','wasn','so','is','isn',
        'it','itself','too','couldn','mustn','i','if','same','her','how','will','can','then',
        'that','these','and','been','have','in','the','a','an'
    ]

# === CONFIG ===

# Now matches up to 4 levels (e.g. "8", "8.5", "8.5.1", "8.5.1.10")
SECTION_PATTERN = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z\(\:][^\n\.]{5,})",
    re.MULTILINE
)  # updated from previous 3-level pattern :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

SWS_ID_PATTERN = re.compile(r"\[SWS_([A-Za-z0-9]+)_(\d{5})\]")

# Skip entire subtrees when their title contains any of these keywords:
SKIP_SUBSECTION_TREE_IF_TITLE_CONTAINS = [
    "service interface",        # catches 8.5 and all its descendants
    "client-server-interfaces",
    "implementation data types",
    "ports",
    "_{" ,
]

# Major sections to skip entirely (and their children)
SKIP_MAJOR_SECTION_AND_CHILDREN_IF_TITLE_CONTAINS = [
    "document change history",
    "disclaimer",
    "table of contents",
    "requirements traceability",
    "published information",
    "not applicable requirements",
    "configuration specification", 
    "how to read this chapter",    
    "containers and configuration parameters", # If this is consistently a MAJOR section to skip with children
    "container", # Often part of "Containers and configuration parameters"
    # "service interface", # Moved to SKIP_SUBSECTION_TREE_IF_TITLE_CONTAINS
    "configuration", # Broad term, be cautious if used for major sections
    "module configuration",        
    "appendix",                    
    "annex"  ,
]

# Single-section skips
SKIP_THESE_SPECIFIC_SECTION_TITLES = [
    "document change history", "disclaimer", "table of contents",
    "requirements traceability", "published information", "not applicable requirements",
    "how to read this chapter",
    # "service interface", # Now handled by SKIP_SUBSECTION_TREE_IF_TITLE_CONTAINS
    "PrimitiveCfg",
    "containers and configuration parameters", # If it's just one specific section title, not a tree
    "module configuration" ,# If it's just one specific section title, not a tree
]

SKIP_THESE_SECTION_IDS = []  # e.g. ["2"]

MIN_CHUNK_WORD_COUNT = 15
TFIDF_KEYWORD_COUNT  = 5

HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*AUTOSAR\s*$", re.IGNORECASE),
    re.compile(r"Specification of [A-Za-z ]+ Layer", re.IGNORECASE),
    re.compile(r"AUTOSAR CP Release \d+\.\d+\.\d+", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"Document ID\s*\d*:\s*AUTOSAR_SWS_[A-Za-z]+", re.IGNORECASE),
    re.compile(r"-\s*AUTOSAR confidential\s*-", re.IGNORECASE),
]
TOC_HEADER_PATTERN = re.compile(r"^\s*Table of Contents\s*$", re.IGNORECASE | re.MULTILINE)
TOC_LINE_PATTERN   = re.compile(r"^[^\n]+\s*\.{5,}\s*\d+\s*$", re.MULTILINE)

# === HELPERS ===

def clean_text_page_wise(raw):
    if TOC_HEADER_PATTERN.search(raw):
        return ""
    out = []
    for line in raw.split('\n'):
        t = line.strip()
        if not t: continue
        if any(p.search(t) for p in HEADER_FOOTER_PATTERNS): continue
        if TOC_LINE_PATTERN.match(t): continue
        out.append(line)
    return "\n".join(out)

def get_module_from_text(text, fname=""):
    m = SWS_ID_PATTERN.search(text)
    if m: return m.group(1)
    f = fname.lower()
    if "crypto" in f or "csm" in f: return "Csm"
    if "cantp" in f: return "CanTp"
    if "pdur" in f: return "PduR"
    if "canif" in f: return "CanIf"
    return None

def bm25_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

def extract_tfidf_keywords_for_corpus(corpus, top_k=TFIDF_KEYWORD_COUNT):
    if not corpus: return [[] for _ in corpus]
    try:
        vec = TfidfVectorizer(
            stop_words=STOP_WORDS_NLTK,
            lowercase=True,
            max_features=2000,
            ngram_range=(1,2)
        )
        mat = vec.fit_transform(corpus)
        feats = np.array(vec.get_feature_names_out())
        kws = []
        for i in range(mat.shape[0]):
            row = mat[i].toarray().flatten()
            idxs = np.argsort(row)[-top_k:][::-1]
            kws.append([feats[j] for j in idxs if row[j]>0])
        return kws
    except Exception as e:
        print(f"❌ TF-IDF error: {e}")
        return [[] for _ in corpus]

def extract_autosar_chunks(pdf_path):
    print(f"📄 Opening PDF: {pdf_path}")
    filename = os.path.basename(pdf_path)
    # Read & clean all pages
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        page_map = []
        offset = 0
        for pno, page in enumerate(doc):
            raw = page.get_text("text", sort=True)
            cleaned = clean_text_page_wise(raw)
            if cleaned.strip():
                full_text += cleaned + "\n\n"
                page_map.append((offset, pno+1))
                offset += len(cleaned) + 2
        doc.close()
    except Exception as e:
        print(f"❌ PDF read error: {e}")
        return []

    # Find section headers
    matches = []
    for m in SECTION_PATTERN.finditer(full_text):
        sec_id    = m.group(1)
        sec_title = m.group(2).strip()
        # ─── Skip “enumeration” entries like “1 E_NOT_OK” or “2 CSM_E_BUSY” ───
        # only uppercase letters, digits, and underscores => not a real section
        if re.fullmatch(r"[A-Z0-9_]+", sec_title):
            continue

        matches.append({
            "id_str":   sec_id,
            "title":    sec_title,
            "start":    m.start(),
            "end":      m.end()
        })

    matches.sort(key=lambda x: x["start"])

    chunks = []
    major_skip = None
    subtree_skip = None

    for i, info in enumerate(matches):
        sid   = info["id_str"]
        title = info["title"]
        print(f"DEBUG: saw section [{sid}] – “{title}”")    # ← add this
        low   = title.lower()
        is_major = "." not in sid

        # Reset logic
        if major_skip is not None:
            if int(sid.split(".")[0]) > major_skip:
                major_skip = None
        if subtree_skip and not (sid == subtree_skip or sid.startswith(subtree_skip + ".")):
            print(f"🔄 Exited skip subtree {subtree_skip}; now at {sid}")
            subtree_skip = None

        skip = False
        # Active skips
        if major_skip and int(sid.split(".")[0]) == major_skip:
            skip = True
        if not skip and subtree_skip:
            skip = True

        # Trigger major skip
        if not skip and is_major and any(k in low for k in SKIP_MAJOR_SECTION_AND_CHILDREN_IF_TITLE_CONTAINS):
            major_skip = int(sid)
            print(f"🔀 Skipping major {sid} '{title}' + children")
            skip = True

        # Trigger subtree skip
        if not skip and any(k in low for k in SKIP_SUBSECTION_TREE_IF_TITLE_CONTAINS):
            subtree_skip = sid
            print(f"⛔ Skipping subtree {sid} '{title}'")
            skip = True

        # Single-title or ID skips
        if not skip and title.lower() in SKIP_THESE_SPECIFIC_SECTION_TITLES:
            print(f"⛔ Skipping specific title {sid} '{title}'")
            skip = True
        if not skip and sid in SKIP_THESE_SECTION_IDS:
            print(f"⛔ Skipping by ID {sid} '{title}'")
            skip = True

        if skip:
            continue

        # Extract chunk text
        start = info["end"]
        end   = matches[i+1]["start"] if i+1 < len(matches) else len(full_text)
        body  = full_text[start:end].strip()
        if len(body.split()) < MIN_CHUNK_WORD_COUNT:
            continue

        # Map to page
        pno = 1
        for offs, pn in reversed(page_map):
            if offs <= start:
                pno = pn
                break

        chunks.append({
            "text":           body,
            "section_id_str": sid,
            "section_title":  title,
            "module":         get_module_from_text(body, filename),
            "page_number":    pno
        })

    print(f"🧩 Prepared {len(chunks)} chunks from {filename}")
    return chunks
