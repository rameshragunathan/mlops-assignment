"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    q_text = question["question"]
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Run gold SQL to get expected rows
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)
    if not gold_ok:
        return {
            "question": q_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "error": f"Gold SQL failed: {gold_err}",
            "iterations": 0,
            "per_iteration_correct": {},
            "final_correct": False,
            "agent_sql": None,
        }

    # Call the agent
    try:
        resp = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "question": q_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "error": f"Agent call failed: {e}",
            "iterations": 0,
            "per_iteration_correct": {},
            "final_correct": False,
            "agent_sql": None,
        }

    agent_sql = data.get("sql", "")
    iterations = data.get("iterations", 0)
    history = data.get("history", [])

    # Build per-iteration correctness
    # history entries are in order: generate_sql, revise, revise, ...
    # We check the SQL at each iteration step
    per_iteration_correct: dict[int, bool] = {}
    for idx, entry in enumerate(history):
        sql = entry.get("sql", "")
        ok, rows, _ = run_sql(db_id, sql)
        correct = matches(gold_rows, rows) if ok else False
        per_iteration_correct[idx + 1] = correct

    # Carry forward: fill up to MAX_ITERATIONS using last known result
    max_iter = max(per_iteration_correct.keys()) if per_iteration_correct else 0
    last = per_iteration_correct.get(max_iter, False)
    for i in range(max_iter + 1, 4):
        per_iteration_correct[i] = last

    final_correct = per_iteration_correct.get(iterations, last)

    return {
        "question": q_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "iterations": iterations,
        "per_iteration_correct": per_iteration_correct,
        "final_correct": final_correct,
        "error": data.get("error"),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results with per-iteration pass rates."""
    total = len(results)
    if total == 0:
        return {"total": 0, "overall_pass_rate": 0.0, "per_iteration_pass_rate": {}}

    # Per-iteration pass rates
    per_iter: dict[int, int] = {1: 0, 2: 0, 3: 0}
    final_correct = 0

    for r in results:
        pic = r.get("per_iteration_correct", {})
        for it in [1, 2, 3]:
            if pic.get(it, False):
                per_iter[it] += 1
        if r.get("final_correct", False):
            final_correct += 1

    return {
        "total": total,
        "overall_pass_rate": round(final_correct / total, 3),
        "per_iteration_pass_rate": {
            str(it): round(per_iter[it] / total, 3)
            for it in [1, 2, 3]
        },
        "final_correct": final_correct,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
