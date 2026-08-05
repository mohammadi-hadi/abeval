"""Command-line interface for abeval."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .ci import clustered_mean_ci, mean_ci, proportion_ci
from .compare import paired_compare
from .data import extract, pair, read_jsonl
from .power import mde, sample_size, sd_diff_from_rates
from .reliability import judge_reliability


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _is_binary(values: list[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values)


def cmd_ci(args: argparse.Namespace) -> int:
    records = read_jsonl(args.results)
    values_by_id = extract(records, args.metric, args.id_key)
    values = list(values_by_id.values())
    if _is_binary(values) and not args.bootstrap:
        interval = proportion_ci(int(sum(values)), len(values), args.level)
    elif args.cluster_key:
        clusters = [rec[args.cluster_key] for rec in records]
        interval = clustered_mean_ci(values, clusters, args.level)
    elif args.bootstrap:
        interval = mean_ci(values, args.level, method="bootstrap", reps=args.reps, seed=args.seed)
    else:
        interval = mean_ci(values, args.level, method="t")
    if args.json:
        print(json.dumps(interval.as_dict(), indent=2))
        return 0
    pct = _is_binary(values)
    fmt = _fmt_pct if pct else lambda v: f"{v:.4g}"
    print(f"{args.metric}: {fmt(interval.estimate)}  "
          f"[{fmt(interval.lo)}, {fmt(interval.hi)}]  "
          f"(n={interval.n}, {int(interval.level * 100)}% CI, {interval.method})")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    rec_a = read_jsonl(args.a)
    rec_b = read_jsonl(args.b)
    map_a = extract(rec_a, args.metric, args.id_key)
    map_b = extract(rec_b, args.metric, args.id_key)
    values_a, values_b, dropped = pair(map_a, map_b)
    if not values_a:
        print("error: no shared item ids between the two runs", file=sys.stderr)
        return 2
    if dropped:
        print(f"note: {dropped} unpaired item(s) dropped", file=sys.stderr)
    result = paired_compare(values_a, values_b, args.level, args.reps, args.seed)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    pct = result.binary
    fmt = _fmt_pct if pct else lambda v: f"{v:+.4g}"
    plain = _fmt_pct if pct else lambda v: f"{v:.4g}"
    sign = "+" if result.diff >= 0 else ""
    print(f"A: {plain(result.mean_a)}   B: {plain(result.mean_b)}   (n={result.n} paired items)")
    print(f"B - A: {sign}{fmt(result.diff).lstrip('+') if not pct else fmt(result.diff)}  "
          f"[{fmt(result.ci_lo)}, {fmt(result.ci_hi)}]  ({int(result.level * 100)}% CI)")
    print(f"p (sign-flip permutation): {result.p_permutation:.4f}   p (paired t): {result.p_t:.4f}")
    if result.mcnemar is not None:
        mc = result.mcnemar
        print(f"discordant items: B-only wins {mc['b_only']}, A-only wins {mc['a_only']}   "
              f"p (exact McNemar): {mc['p']:.4f}")
    verdict = (
        "significant at the chosen level"
        if result.p_permutation < 1.0 - args.level
        else "NOT significant — the difference is within noise at this sample size"
    )
    print(f"verdict: {verdict}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    sd = sd_diff_from_rates(args.baseline, args.baseline + args.delta, args.corr)
    n = sample_size(args.delta, sd, args.level, args.power)
    out = {
        "baseline": args.baseline,
        "delta": args.delta,
        "corr": args.corr,
        "sd_diff": sd,
        "level": args.level,
        "power": args.power,
        "n_items": n,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"to detect {_fmt_pct(abs(args.delta))} difference from a "
          f"{_fmt_pct(args.baseline)} baseline (corr={args.corr}):")
    print(f"  n = {n} paired items  "
          f"({int(args.power * 100)}% power, {int(args.level * 100)}% confidence)")
    for n_have in (100, 200, 500, 1000):
        d = mde(n_have, sd, args.level, args.power)
        print(f"  with {n_have:>4} items you can detect >= {_fmt_pct(d)}")
    return 0


def cmd_reliability(args: argparse.Namespace) -> int:
    records = read_jsonl(args.results)
    groups: dict = {}
    for rec in records:
        if args.id_key not in rec or args.metric not in rec:
            raise ValueError(f"record missing {args.id_key!r} or {args.metric!r}: {rec}")
        groups.setdefault(rec[args.id_key], []).append(rec[args.metric])
    rel = judge_reliability(groups)
    if args.json:
        print(json.dumps(rel.as_dict(), indent=2))
        return 0
    print(f"{rel.n_items} items, {rel.n_judgments} judgments "
          f"({rel.mean_reps:.1f} per item)")
    print(f"ICC (signal share of variance): {rel.icc:.3f}")
    print(f"between-item sd: {rel.sd_between:.4g}   judge noise sd: {rel.sd_within:.4g}")
    print(f"exact agreement across repeats: {_fmt_pct(rel.exact_agreement)}")
    reps = rel.repeats_for(0.1)
    if reps is not None:
        print(f"repeats per item to push judge noise under 10%: {reps}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="abeval",
        description="A/B-test statistics for LLM evals.",
    )
    parser.add_argument("--version", action="version", version=f"abeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--metric", default="score", help="metric field name (default: score)")
    common.add_argument("--id-key", default="id", help="item id field name (default: id)")
    common.add_argument("--level", type=float, default=0.95, help="confidence level (default: 0.95)")
    common.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    common.add_argument("--reps", type=int, default=10_000,
                        help="bootstrap/permutation resamples (default: 10000)")
    common.add_argument("--json", action="store_true", help="emit JSON instead of text")

    p_ci = sub.add_parser("ci", parents=[common],
                          help="confidence interval for one run's metric")
    p_ci.add_argument("results", help="JSONL file with per-item results")
    p_ci.add_argument("--bootstrap", action="store_true",
                      help="percentile bootstrap instead of t/Wilson")
    p_ci.add_argument("--cluster-key", default=None,
                      help="field defining dependence clusters (e.g. source document)")
    p_ci.set_defaults(fn=cmd_ci)

    p_cmp = sub.add_parser("compare", parents=[common],
                           help="paired comparison of two runs on shared items")
    p_cmp.add_argument("a", help="baseline run JSONL")
    p_cmp.add_argument("b", help="candidate run JSONL")
    p_cmp.set_defaults(fn=cmd_compare)

    p_pow = sub.add_parser("power", parents=[common],
                           help="how many items you need before you run")
    p_pow.add_argument("--baseline", type=float, required=True,
                       help="expected baseline accuracy, e.g. 0.75")
    p_pow.add_argument("--delta", type=float, required=True,
                       help="difference you care about detecting, e.g. 0.03")
    p_pow.add_argument("--corr", type=float, default=0.5,
                       help="item-level correlation between runs (default: 0.5)")
    p_pow.add_argument("--power", type=float, default=0.8,
                       help="target power (default: 0.8)")
    p_pow.set_defaults(fn=cmd_power)

    p_rel = sub.add_parser("reliability", parents=[common],
                           help="judge-noise decomposition from repeated judgments")
    p_rel.add_argument("results", help="JSONL with repeated judgments per item id")
    p_rel.set_defaults(fn=cmd_reliability)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
