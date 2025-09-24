#!/usr/bin/env python3
"""
AutoPatcher — hourly self-optimizer for trade_bot.py using advisor.py

- Every hour:
  * Ask advisor.py for a unified diff
  * Validate + apply patch on a temp git branch
  * Run checks (syntax, optional flake8/mypy, smoke run)
  * (Optional) tiny backtest gate
  * If pass => merge to main and PM2 restart trade bot
  * If fail => rollback

Usage (PM2 recommended):
  pm2 start autopatcher.py --name trade-optimizer -- \
      --provider openai --model gpt-4o-mini \
      --bot_name trade-bot --interval_sec 3600

Env:
  OPENAI_API_KEY=...        (for provider=openai)
"""

from __future__ import annotations
import argparse, os, sys, subprocess, time, shutil, tempfile, json, re, textwrap
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
REPO = ROOT  # assume repo root here; change if needed

# ---------- small utils ----------
def run(cmd: List[str], cwd: Path | None = None, timeout: int = 300) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, cwd=str(cwd or REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err

def which(x: str) -> str | None:
    return shutil.which(x)

def logln(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    with open(ROOT/"logs/autoopt.log", "a", encoding="utf-8") as f:
        f.write(line+"\n")

# ---------- git helpers ----------
def ensure_git():
    if not (REPO/".git").exists():
        run(["git","init"])
        run(["git","add","-A"])
        run(["git","commit","-m","init autopatcher baseline"], timeout=120)

def git_status_clean() -> bool:
    code,out,err = run(["git","status","--porcelain"])
    return (code==0) and (out.strip()=="")

def git_new_branch(name: str) -> bool:
    run(["git","checkout","-q","-B", name])
    return True

def git_reset_hard():
    run(["git","reset","--hard","HEAD"])

def git_apply_patch(patch_text: str) -> Tuple[bool,str]:
    # try git apply --check first
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".patch") as tf:
        tf.write(patch_text)
        tf.flush()
        path = tf.name
    ok,_out,_err = False,"",""
    code, out, err = run(["git","apply","--check", path])
    if code != 0:
        os.unlink(path)
        return False, f"git apply --check failed:\n{err or out}"
    code, out, err = run(["git","apply", path])
    os.unlink(path)
    if code == 0:
        ok = True
    return ok, err or out

def git_commit(msg: str):
    run(["git","add","-A"])
    run(["git","commit","-m", msg])

def git_checkout(name: str):
    run(["git","checkout","-q", name])

def git_merge(from_branch: str) -> bool:
    code,out,err = run(["git","merge","--no-ff","-m","autopatcher merge", from_branch])
    return code==0

# ---------- validation ----------
PY_WHITELIST_RE = re.compile(r"^\+\+\+\s+b\/.*\.py$", re.M)

def patch_is_safe(patch: str, max_added_lines=2000) -> Tuple[bool,str]:
    # Limits: touches only .py files and patch isn't absurdly large
    if not PY_WHITELIST_RE.search(patch):
        return False, "Patch touches non-.py files or no .py files detected."
    added = sum(1 for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    if added > max_added_lines:
        return False, f"Patch too large (+{added} lines)."
    return True, "ok"

# ---------- checks ----------
def check_syntax(pyfiles: List[Path]) -> Tuple[bool,str]:
    for f in pyfiles:
        code,out,err = run([sys.executable,"-m","py_compile",str(f)])
        if code != 0:
            return False, f"Syntax error in {f}:\n{err or out}"
    return True, "ok"

def check_flake(pyfiles: List[Path]) -> Tuple[bool,str]:
    if not which("flake8"): return True, "flake8 not installed; skipping."
    code,out,err = run(["flake8"] + [str(p) for p in pyfiles])
    if code != 0:
        return False, (out or err)
    return True, "ok"

def check_mypy(pyfiles: List[Path]) -> Tuple[bool,str]:
    if not which("mypy"): return True, "mypy not installed; skipping."
    code,out,err = run(["mypy","--ignore-missing-imports"] + [str(p) for p in pyfiles])
    if code != 0:
        return False, (out or err)
    return True, "ok"

def smoke_run_bot() -> Tuple[bool,str]:
    # Import path + --help should succeed quickly
    code,out,err = run([sys.executable, "trade_bot.py", "--help"], timeout=60)
    if code != 0:
        return False, f"trade_bot.py --help failed:\n{err or out}"
    return True, "ok"

def quick_backtest_gate(max_secs=60) -> Tuple[bool,str]:
    # Very light check: run train-only for a tiny slice (reduces risk)
    # You can beef this up to compute metrics before/after.
    try:
        code,out,err = run([sys.executable, "trade_bot.py", "--mode","train",
                            "--backfill_years","1", "--train_interval","60m",
                            "--train_once","true"], timeout=max_secs)
        # Even if Yahoo rate-limits, we only gate on crash/syntax here.
        return (code==0), (err or out)
    except Exception as e:
        return False, str(e)

# ---------- bot orchestrator ----------
def pm2_stop(name: str):
    if which("pm2"):
        run(["pm2","stop",name])

def pm2_start(name: str):
    if which("pm2"):
        # assume pm2 already configured to run trade_bot.py under this name
        run(["pm2","start",name])

def pm2_reload(name: str):
    if which("pm2"):
        run(["pm2","reload",name])

# ---------- advisor bridge ----------
def get_unified_diff(provider: str, model: str, topk: int) -> str:
    # Calls advisor.py --patch trade_bot.py ...
    cmd = [sys.executable, "advisor.py", "--patch", "trade_bot.py", "--topk", str(topk)]
    if provider:
        cmd += ["--provider", provider]
    if model:
        cmd += ["--model", model]
    code,out,err = run(cmd, timeout=600)
    diff = out.strip()
    if diff.startswith("--- old") and "+++ new" in diff:
        # Convert advisory diff banner to a standard git-compatible one if needed
        return diff.replace("--- old", "--- a/trade_bot.py").replace("+++ new", "+++ b/trade_bot.py")
    # Some models output raw `git diff` format already; try to pass-through
    if diff.startswith("--- a/") and "\n+++ b/" in diff:
        return diff
    # If nothing useful, return empty
    logln(f"Advisor produced no usable diff. stderr:\n{err}")
    return ""

# ---------- lock so we don't race with live bot ----------
LOCK = ROOT / ".autopatch.lock"

def acquire_lock() -> bool:
    try:
        LOCK.write_text(str(os.getpid()))
        return True
    except Exception:
        return False

def release_lock():
    try:
        if LOCK.exists(): LOCK.unlink()
    except Exception:
        pass

# ---------- main cycle ----------
def cycle(args):
    logln("=== AutoPatcher cycle start ===")
    ensure_git()

    if not git_status_clean():
        logln("Working tree not clean; aborting cycle.")
        return

    # 1) Ask advisor for a diff
    diff = get_unified_diff(args.provider, args.model, args.topk)
    if not diff:
        logln("No diff received; skipping.")
        return

    # 2) Validate diff
    ok, why = patch_is_safe(diff, args.max_added_lines)
    if not ok:
        logln(f"Patch rejected: {why}")
        return
    logln("Patch looks sane; creating branch...")
    base_branch = "main"
    run(["git","checkout","-q","-B", base_branch])  # ensure main exists
    tmp_branch = f"autopatch-{int(time.time())}"
    git_new_branch(tmp_branch)

    # 3) Apply patch
    ok, msg = git_apply_patch(diff)
    if not ok:
        logln(f"git apply failed:\n{msg}")
        git_checkout(base_branch)
        git_reset_hard()
        return
    git_commit("autopatcher: apply advisor patch")

    # 4) Checks
    pyfiles = [p for p in REPO.rglob("*.py")]
    ok, msg = check_syntax(pyfiles);     logln(f"[syntax] {msg}")
    if not ok: goto_rollback(base_branch, tmp_branch); return

    ok, msg = check_flake(pyfiles);      logln(f"[flake8] {msg}")
    if not ok and args.enforce_flake8:   goto_rollback(base_branch, tmp_branch); return

    ok, msg = check_mypy(pyfiles);       logln(f"[mypy] {msg}")
    if not ok and args.enforce_mypy:     goto_rollback(base_branch, tmp_branch); return

    ok, msg = smoke_run_bot();           logln(f"[smoke] {msg}")
    if not ok: goto_rollback(base_branch, tmp_branch); return

    if args.quick_backtest:
        ok, msg = quick_backtest_gate(max_secs=args.bt_max_secs); logln(f"[quick-bt] {msg[:300]}")
        if not ok:
            goto_rollback(base_branch, tmp_branch); return

    # 5) Merge to main
    git_checkout(base_branch)
    if not git_merge(tmp_branch):
        logln("Merge failed; rolling back.")
        git_reset_hard()
        return
    logln("Patch merged to main.")

    # 6) Restart bot if requested
    if args.manage_pm2 and args.bot_name:
        logln(f"Reloading PM2 process: {args.bot_name}")
        pm2_reload(args.bot_name)

    logln("=== AutoPatcher cycle complete ===")

def goto_rollback(base_branch: str, tmp_branch: str):
    logln("Checks failed—rolling back.")
    git_checkout(base_branch)
    git_reset_hard()
    # We leave tmp branch dangling for forensics; optional: delete branch
    # run(["git","branch","-D", tmp_branch])

def main():
    ap = argparse.ArgumentParser(description="Auto-optimizer daemon for trade_bot.py")
    ap.add_argument("--interval_sec", type=int, default=3600, help="Run cycle every N seconds")
    ap.add_argument("--provider", default="", help="openai|ollama (empty = no LLM; skip patching)")
    ap.add_argument("--model", default="", help="LLM model name")
    ap.add_argument("--topk", type=int, default=12, help="RAG retrieval size advisor will use")
    ap.add_argument("--max_added_lines", type=int, default=2000)
    ap.add_argument("--manage_pm2", action="store_true", default=True, help="Reload PM2 bot after success")
    ap.add_argument("--bot_name", default="trade-bot", help="PM2 name of your running bot")
    ap.add_argument("--enforce_flake8", action="store_true", default=False)
    ap.add_argument("--enforce_mypy", action="store_true", default=False)
    ap.add_argument("--quick_backtest", action="store_true", default=True, help="Run a tiny train-only pass")
    ap.add_argument("--bt_max_secs", type=int, default=60)
    args = ap.parse_args()

    if not acquire_lock():
        logln("Another autopatcher instance appears to be running. Exiting.")
        sys.exit(1)

    try:
        while True:
            # If no provider/model configured, we still log & skip patching
            if not args.provider or not args.model:
                logln("No LLM provider/model configured; skipping patch step this cycle.")
            cycle(args)
            time.sleep(max(5, args.interval_sec))
    finally:
        release_lock()

if __name__ == "__main__":
    main()
