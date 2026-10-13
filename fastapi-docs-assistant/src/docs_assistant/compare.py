"""Compare two eval runs on the same items, with paired bootstrap CIs.

Run:  python -m docs_assistant.compare evals/runs/<baseline> evals/runs/<candidate>

Only items present in both runs are compared, item by item, so per-item
difficulty cancels out. A difference is called significant only when its 95%
CI excludes zero; with ~40 items, small changes usually won't be.
"""
import argparse
from pathlib import Path

from .eval_metrics import paired_bootstrap
from .evaluate import load_run

CONFIG_KEYS = ("chat_model", "judge_model", "prompt_version", "judge_prompt_version", "k", "collection",
               "embedding_model", "chunk_max_tokens", "corpus_commit", "golden_sha256", "split")


def metrics(row: dict, k: int) -> dict:
    ret = row.get("retrieval") or {}
    nug = row.get("nuggets") or {}
    return {
        "pass": float(row["passed"]),
        "hit@1": float(ret["hit@1"]) if ret else None,
        f"hit@{k}": float(ret[f"hit@{k}"]) if ret else None,
        "mrr": ret.get("reciprocal_rank") if ret else None,
        "vital_nugget_recall": nug.get("vital_strict"),
        "false_refusal": float(row["false_refusal"]) if row.get("false_refusal") is not None else None,
    }


def compare(run_a: tuple[dict, list[dict]], run_b: tuple[dict, list[dict]], resamples: int = 5000) -> dict:
    (cfg_a, rows_a), (cfg_b, rows_b) = run_a, run_b
    k = cfg_a["k"]
    a_by_id = {r["id"]: r for r in rows_a if not r.get("error")}
    b_by_id = {r["id"]: r for r in rows_b if not r.get("error")}
    shared = sorted(a_by_id.keys() & b_by_id.keys())
    ma = {i: metrics(a_by_id[i], k) for i in shared}
    mb = {i: metrics(b_by_id[i], k) for i in shared}

    deltas = {}
    for name in next(iter(ma.values()), {}):
        deltas[name] = paired_bootstrap([ma[i][name] for i in shared], [mb[i][name] for i in shared],
                                        resamples=resamples)
    return {
        "n_shared": len(shared),
        "config_changes": {key: (cfg_a.get(key), cfg_b.get(key)) for key in CONFIG_KEYS
                           if cfg_a.get(key) != cfg_b.get(key)},
        "deltas": deltas,
        "fixed": [i for i in shared if not a_by_id[i]["passed"] and b_by_id[i]["passed"]],
        "broken": [i for i in shared if a_by_id[i]["passed"] and not b_by_id[i]["passed"]],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two eval runs item by item.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    a, b = load_run(args.baseline), load_run(args.candidate)
    result = compare(a, b)
    print(f"{a[0]['label']} -> {b[0]['label']}   ({result['n_shared']} shared items)\n")
    if "golden_sha256" in result["config_changes"] or "corpus_commit" in result["config_changes"]:
        print("WARNING: the runs use a different golden set or corpus; differences mix several causes.\n")
    print("Changed settings:" if result["config_changes"] else "No setting changes recorded.")
    for key, (old, new) in result["config_changes"].items():
        print(f"  {key}: {old} -> {new}")
    print(f"\n{'metric':<22}{'change':>10}   95% CI              significant")
    for name, d in result["deltas"].items():
        if d is None:
            continue
        lo, hi = d["ci"]
        print(f"{name:<22}{d['diff']:>+10.3f}   [{lo:+.3f}, {hi:+.3f}]   {'yes' if d['significant'] else 'no'}"
              f"   (n={d['n']})")
    print(f"\nFixed  ({len(result['fixed'])}): {', '.join(result['fixed']) or '-'}")
    print(f"Broken ({len(result['broken'])}): {', '.join(result['broken']) or '-'}")


if __name__ == "__main__":
    main()
