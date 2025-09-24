#!/usr/bin/env python3
"""
DevAdvisor — a local "secondary AI" for Python-stdlib-aware RAG + code review.

Features
- Build a local KB from Python stdlib docs (pydoc) + your repo's .py files (no internet).
- Vector search via TF-IDF + TruncatedSVD (fast, zero external model). Optional sentence-transformers.
- Static checks (flake8/mypy if available), quick cProfile run, rough latency micro-bench hooks.
- LLM plug-in (OpenAI or Ollama) to produce critiques and unified diffs. Works without keys (it will just skip LLM calls).

Usage:
  python3 advisor.py --build
  python3 advisor.py --review trade_bot.py
  python3 advisor.py --patch trade_bot.py --provider openai --model gpt-4o-mini
  python3 advisor.py --bench trade_bot.py
"""

from __future__ import annotations
import argparse, os, sys, json, time, pkgutil, inspect, importlib, traceback, subprocess, textwrap, re, io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# --- Optional heavy deps handled gracefully ---
try:
    from sentence_transformers import SentenceTransformer  # optional
    _HAS_ST = True
except Exception:
    _HAS_ST = False

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

import pydoc
import sysconfig

INDEX_DIR = Path("./advisor_index")
DOCS_PKL  = INDEX_DIR / "docs.pkl"
CODE_PKL  = INDEX_DIR / "code.pkl"
VEC_PKL   = INDEX_DIR / "vec.pkl"
SVD_PKL   = INDEX_DIR / "svd.pkl"
NN_PKL    = INDEX_DIR / "nn.pkl"
META_PKL  = INDEX_DIR / "meta.json"

# ---- Simple pickle helpers (no external deps) ----
import pickle
def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: pickle.dump(obj, f)

def load(path: Path):
    with open(path, "rb") as f: return pickle.load(f)

# ---------------------- Indexing ----------------------
def iter_stdlib_modules() -> List[str]:
    stdlib_path = sysconfig.get_paths().get("stdlib", "")
    mods = []
    for m in pkgutil.walk_packages([stdlib_path]):
        name = m.name
        # Skip test packages and private modules to speed up
        if name.startswith("_") or ".tests" in name or name.endswith(".__main__"):
            continue
        # Use top-level names (pydoc handles sub-members)
        mods.append(name)
    # Some crucial well-knowns (ensure present)
    extras = ["asyncio","concurrent","multiprocessing","threading","logging","pathlib","functools",
              "itertools","collections","statistics","subprocess","datetime","time","json","re","io",
              "unittest","importlib","queue","heapq","bisect","math","random","traceback"]
    for e in extras:
        if e not in mods: mods.append(e)
    return sorted(set(mods))[:1200]  # cap for speed

def doc_of_module(modname: str) -> str:
    try:
        # Don't import arbitrary third-party packages; stdlib only.
        # Let pydoc import; if it fails, skip.
        txt = pydoc.render_doc(modname, "Help on %s")
        return txt
    except Exception:
        return ""

def chunk_text(text: str, tokens: int = 800, overlap: int = 120) -> List[str]:
    # crude line-based chunking approximating token windows
    lines = text.splitlines()
    chunks, buf = [], []
    count = 0
    for ln in lines:
        ln = ln.strip("\n")
        l = max(1, len(ln)//4)  # rough token-ish heuristic
        if count + l > tokens and buf:
            chunks.append("\n".join(buf))
            if overlap > 0:
                buf = buf[-overlap:]
                count = sum(max(1, len(x)//4) for x in buf)
            else:
                buf = []; count = 0
        buf.append(ln); count += l
    if buf: chunks.append("\n".join(buf))
    return chunks

def gather_code_chunks(root: Path) -> List[Tuple[str,str]]:
    files = list(root.rglob("*.py"))
    results = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            # Split by top-level defs/classes or by size
            parts = re.split(r"(?m)^(def |class )", text)
            if len(parts) > 1:
                # Re-stitch with headers
                stitched = [parts[0]]
                for i in range(1, len(parts), 2):
                    stitched.append(parts[i] + parts[i+1] if i+1 < len(parts) else parts[i])
                blocks = stitched
            else:
                blocks = chunk_text(text, tokens=800, overlap=120)
            for i, blk in enumerate(blocks):
                if blk.strip():
                    results.append((f"{fp}::chunk{i}", blk))
        except Exception:
            continue
    return results

@dataclass
class KB:
    docs_texts: List[str]
    docs_meta:   List[Dict]
    code_texts: List[str]
    code_meta:  List[Dict]

def build_index(project_root: Path) -> None:
    print("[Advisor] Building KB (stdlib docs + repo code)...")
    # --- stdlib docs
    mods = iter_stdlib_modules()
    docs_texts, docs_meta = [], []
    for i, m in enumerate(mods, 1):
        t = doc_of_module(m)
        if not t: continue
        for j, ch in enumerate(chunk_text(t, tokens=900, overlap=150)):
            docs_texts.append(ch)
            docs_meta.append({"source": f"stdlib:{m}", "chunk": j})
        if i % 50 == 0:
            print(f"  indexed stdlib modules: {i}/{len(mods)}")

    # --- repo code
    code_pairs = gather_code_chunks(project_root)
    code_texts = [c for _, c in code_pairs]
    code_meta  = [{"source": k, "chunk": idx} for idx, (k, _) in enumerate(code_pairs)]

    kb = KB(docs_texts, docs_meta, code_texts, code_meta)
    dump(kb, DOCS_PKL)  # store together
    dump(kb, CODE_PKL)
    print(f"[Advisor] Indexed {len(docs_texts)} doc chunks and {len(code_texts)} code chunks.")

    # --- Vectorizer (default TF-IDF + SVD) ---
    all_texts = docs_texts + code_texts
    print("[Advisor] Fitting TF-IDF + SVD...")
    tfidf = TfidfVectorizer(
        lowercase=True, max_df=0.9, min_df=2, ngram_range=(1,2),
        stop_words="english"
    ).fit(all_texts)
    X = tfidf.transform(all_texts)
    k = min(256, min(X.shape)-1) if min(X.shape) > 1 else 64
    svd = TruncatedSVD(n_components=k, random_state=42).fit(X)
    Xs = svd.transform(X)
    nn = NearestNeighbors(n_neighbors=8, metric="cosine").fit(Xs)

    dump(tfidf, VEC_PKL); dump(svd, SVD_PKL); dump(nn, NN_PKL)
    meta = {"counts": {"docs": len(docs_texts), "code": len(code_texts)}}
    META_PKL.write_text(json.dumps(meta, indent=2))
    print("[Advisor] Done.")

def _ensure_index():
    for p in [DOCS_PKL, CODE_PKL, VEC_PKL, SVD_PKL, NN_PKL, META_PKL]:
        if not p.exists():
            print("[Advisor] KB missing. Run: python3 advisor.py --build")
            sys.exit(1)

def knn_search(query: str, topk=10) -> List[Tuple[str,Dict,float]]:
    _ensure_index()
    kb = load(DOCS_PKL)  # contains KB
    tfidf = load(VEC_PKL); svd = load(SVD_PKL); nn = load(NN_PKL)

    all_texts = kb.docs_texts + kb.code_texts
    all_meta  = kb.docs_meta  + kb.code_meta

    qv = svd.transform(tfidf.transform([query]))
    dists, idxs = nn.kneighbors(qv, n_neighbors=min(topk, len(all_texts)))
    out = []
    for i, d in zip(idxs[0], dists[0]):
        out.append((all_texts[i], all_meta[i], float(1 - d)))  # similarity ~ (1 - distance)
    return out

# ---------------------- Static analysis / Bench ----------------------
def run_cmd(cmd: List[str]) -> Tuple[int,str,str]:
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=120)
        return p.returncode, out, err
    except Exception as e:
        return 1, "", str(e)

def static_checks(pyfile: str) -> Dict[str,str]:
    res = {}
    if shutil_which("flake8"):
        code,out,err = run_cmd(["flake8", pyfile])
        res["flake8"] = out or err or "ok"
    if shutil_which("mypy"):
        code,out,err = run_cmd(["mypy", "--ignore-missing-imports", pyfile])
        res["mypy"] = out or err or "ok"
    return res

def shutil_which(x: str) -> Optional[str]:
    from shutil import which
    return which(x)

def quick_profile(pyfile: str) -> str:
    try:
        import cProfile, pstats
        pr = cProfile.Profile()
        ns = {}
        code = Path(pyfile).read_text(encoding="utf-8", errors="ignore")
        pr.enable()
        exec(compile(code, pyfile, "exec"), ns, ns)  # imports/defines only
        pr.disable()
        s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(30)
        return s.getvalue()
    except SystemExit:
        return "Program called sys.exit(); profiling limited to import path."
    except Exception as e:
        return f"Profile error: {e}\n{traceback.format_exc()}"

def micro_bench(pyfile: str, reps=3) -> str:
    start = time.time()
    errs = []
    for i in range(reps):
        try:
            run_cmd([sys.executable, pyfile, "--help"])
        except Exception as e:
            errs.append(str(e))
    dur = time.time() - start
    return f"Ran --help {reps}x in {dur:.2f}s. Errors: {len(errs)}"

# ---------------------- LLM plumbing (optional) ----------------------
def llm_chat(messages: List[Dict], provider: str, model: str, temperature=0.2, max_tokens=1400) -> str:
    provider = (provider or "").lower()
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return "[LLM] OPENAI_API_KEY not set. Skipping."
        try:
            import openai
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM] OpenAI error: {e}"
    elif provider == "ollama":
        try:
            import requests
            r = requests.post("http://localhost:11434/api/chat", json={
                "model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}
            }, timeout=300)
            if r.ok:
                j = r.json()
                return j.get("message", {}).get("content", "")
            return f"[LLM] Ollama HTTP {r.status_code} {r.text[:120]}"
        except Exception as e:
            return f"[LLM] Ollama error: {e}"
    return "[LLM] No provider configured. Use --provider openai|ollama."

def make_review_prompt(pytext: str, kn_chunks: List[Tuple[str,Dict,float]]) -> List[Dict]:
    context = []
    for txt, meta, score in kn_chunks[:10]:
        src = meta.get("source","?")
        context.append(f"[{src} | sim={score:.2f}]\n{txt}")
    ctx = "\n\n".join(context)
    sys_prompt = (
        "You are a senior Python performance engineer and ML infra reviewer. "
        "Given the user's code and retrieved Python stdlib docs, identify concrete improvements "
        "for performance, concurrency, memory, numerical stability, and maintainability. "
        "Prefer actionable diffs and micro-optimizations that matter in a minute-bar polling loop."
    )
    user_prompt = f"CODE:\n<<<\n{pytext}\n>>>\n\nDOC CONTEXT:\n<<<\n{ctx}\n>>>\n\nReturn a terse, numbered list with code snippets."
    return [{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}]

def make_patch_prompt(pytext: str, kn_chunks: List[Tuple[str,Dict,float]]) -> List[Dict]:
    context = []
    for txt, meta, score in kn_chunks[:12]:
        src = meta.get("source","?")
        context.append(f"[{src} | sim={score:.2f}]\n{txt}")
    ctx = "\n\n".join(context)
    sys_prompt = (
        "You are a refactoring assistant. Produce a unified diff (patch) against the given file. "
        "Keep changes surgical and explain with comments. Do not invent APIs."
    )
    user_prompt = f"FILE CONTENTS:\n<<<\n{pytext}\n>>>\n\nREFERENCE DOCS:\n<<<\n{ctx}\n>>>\n\nReturn only a unified diff starting with '--- old\n+++ new'."
    return [{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}]

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="DevAdvisor (stdlib RAG + code reviewer)")
    ap.add_argument("--build", action="store_true", help="Build KB from stdlib docs + repo code")
    ap.add_argument("--review", metavar="FILE", help="Generate an optimization review")
    ap.add_argument("--patch", metavar="FILE", help="Propose a unified diff (requires LLM)")
    ap.add_argument("--bench", metavar="FILE", help="Quick profile/benchmark")
    ap.add_argument("--project_root", default=".", help="Repo root for code indexing")
    ap.add_argument("--provider", default="", help="LLM provider: openai|ollama")
    ap.add_argument("--model", default="", help="LLM model name")
    ap.add_argument("--topk", type=int, default=12, help="RAG top-k retrieval")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    if args.build:
        build_index(root)
        return

    if args.review:
        _ensure_index()
        pytext = Path(args.review).read_text(encoding="utf-8", errors="ignore")
        q = "optimize high-frequency minute-bar polling, async IO, yfinance, sklearn models, robustness, pm2 logging"
        kn = knn_search(q, topk=args.topk)
        # Try static checks
        checks = static_checks(args.review)
        prof = quick_profile(args.review)
        bench = micro_bench(args.review, reps=2)

        msg = [
            "## Static checks",
            json.dumps(checks, indent=2),
            "\n## Profiling (import phase, top 30 cumtime):\n",
            prof,
            "\n## Micro-bench\n",
            bench,
            "\n## RAG top sources:\n",
            "\n".join([f"- {m['source']} (sim={s:.2f})" for _, m, s in kn[:8]]),
        ]
        print("\n".join(msg))

        # If LLM configured, ask for an expert review
        if args.provider and args.model:
            messages = make_review_prompt(pytext, kn)
            out = llm_chat(messages, provider=args.provider, model=args.model)
            print("\n## LLM Review\n")
            print(out)
        else:
            print("\n[Tip] Add --provider openai --model gpt-4o-mini (or --provider ollama --model llama3.1) for an AI review.")
        return

    if args.patch:
        _ensure_index()
        pytext = Path(args.patch).read_text(encoding="utf-8", errors="ignore")
        if not (args.provider and args.model):
            print("[Advisor] --patch requires --provider and --model (OpenAI or Ollama).")
            sys.exit(1)
        kn = knn_search("refactor for concurrency, safety, and speed; sklearn, pandas, asyncio, logging", topk=args.topk)
        messages = make_patch_prompt(pytext, kn)
        out = llm_chat(messages, provider=args.provider, model=args.model, temperature=0.1, max_tokens=2400)
        print(out)
        return

    if args.bench:
        print(json.dumps(static_checks(args.bench), indent=2))
        print("\n[profile]\n" + quick_profile(args.bench))
        print("\n[bench]\n" + micro_bench(args.bench, reps=3))
        return

    ap.print_help()

if __name__ == "__main__":
    main()
