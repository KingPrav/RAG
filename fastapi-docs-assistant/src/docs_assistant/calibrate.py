"""Measure how well the LLM judge agrees with a human (you).

    python -m docs_assistant.calibrate export evals/runs/<run>    # -> evals/judge_calibration.csv
    (fill in the human_verdict column: pass / fail)
    python -m docs_assistant.calibrate score                      # TPR / TNR / agreement

Following Husain: treat the human labels as ground truth and report the
judge's true positive rate (agrees on passes) and true negative rate (agrees
on fails) separately, since raw agreement hides a judge that passes everything.
"""
import argparse
import csv
import json
from pathlib import Path

from . import config
from .eval_metrics import bootstrap_ci
from .evaluate import load_run

CALIBRATION_PATH = config.ROOT / "evals" / "judge_calibration.csv"
FIELDS = ["id", "category", "question", "reference", "nuggets", "answer", "judge_critique",
          "judge_verdict", "human_verdict", "note"]


def export(run_dir: Path, golden: dict[str, dict], out: Path = CALIBRATION_PATH) -> int:
    cfg, rows = load_run(run_dir)
    judged = [r for r in rows if r.get("judge")]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in judged:
            item = golden[r["id"]]
            w.writerow({
                "id": r["id"], "category": r["category"], "question": r["question"],
                "reference": item["expected_answer"],
                "nuggets": " | ".join(f"[{'VITAL' if n['vital'] else 'okay'}] {n['fact']}" for n in item["nuggets"]),
                "answer": r["answer"], "judge_critique": r["judge"]["critique"],
                "judge_verdict": "pass" if r["passed"] else "fail", "human_verdict": "", "note": "",
            })
    return len(judged)


def score(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("human_verdict", "").strip().lower() in ("pass", "fail")]
    pairs = [(r["human_verdict"].strip().lower() == "pass", r["judge_verdict"].strip().lower() == "pass")
             for r in labeled]
    human_pass = [j for h, j in pairs if h]
    human_fail = [not j for h, j in pairs if not h]
    agree = [float(h == j) for h, j in pairs]
    return {
        "n_labeled": len(pairs), "n_human_pass": len(human_pass), "n_human_fail": len(human_fail),
        "tpr": sum(human_pass) / len(human_pass) if human_pass else None,       # judge passes what you pass
        "tnr": sum(human_fail) / len(human_fail) if human_fail else None,       # judge fails what you fail
        "agreement": sum(agree) / len(agree) if agree else None,
        "agreement_ci95": bootstrap_ci(agree) if len(agree) >= 2 else None,
        "disagreements": [r["id"] for r, (h, j) in zip(labeled, pairs) if h != j],
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against human labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("run_dir", type=Path)
    sub.add_parser("score")
    args = parser.parse_args()

    if args.command == "export":
        from .golden import GOLDEN_PATH
        golden = {json.loads(line)["id"]: json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line}
        n = export(args.run_dir, golden)
        print(f"Wrote {n} judged answers to {CALIBRATION_PATH}. Fill in human_verdict (pass/fail), then run: "
              "python -m docs_assistant.calibrate score")
        return

    with CALIBRATION_PATH.open(newline="") as f:
        s = score(list(csv.DictReader(f)))
    if not s["n_labeled"]:
        raise SystemExit("No human_verdict filled in yet.")
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    print(f"Labeled: {s['n_labeled']} ({s['n_human_pass']} human pass, {s['n_human_fail']} human fail)")
    print(f"True positive rate (judge passes what you pass): {pct(s['tpr'])}")
    print(f"True negative rate (judge fails what you fail):  {pct(s['tnr'])}")
    ci = s["agreement_ci95"]
    print(f"Raw agreement: {pct(s['agreement'])}" + (f" [{pct(ci[0])}, {pct(ci[1])}]" if ci else ""))
    if s["n_human_fail"] < 5:
        print("Note: fewer than 5 human 'fail' labels, so the TNR is not yet meaningful.")
    print(f"Disagreements to inspect: {', '.join(s['disagreements']) or 'none'}")


if __name__ == "__main__":
    main()
