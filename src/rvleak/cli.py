"""Command line interface for rvleak."""

from __future__ import annotations

import argparse
import sys

from . import campaign, report, victims
from .uarch import CacheConfig, ModelConfig


def _model_from_args(args) -> ModelConfig:
    cfg = ModelConfig()
    cfg.dcache = CacheConfig(sets=args.sets, ways=args.ways,
                             line_bytes=args.line, miss_penalty=args.miss_penalty)
    cfg.noise_sigma = args.noise
    cfg.bpred = args.bpred
    cfg.div_data_dependent = args.data_dependent_div
    return cfg


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("microarchitecture")
    g.add_argument("--sets", type=int, default=64, help="D-cache sets")
    g.add_argument("--ways", type=int, default=4, help="D-cache associativity")
    g.add_argument("--line", type=int, default=32, help="cache line size in bytes")
    g.add_argument("--miss-penalty", type=int, default=20, help="cycles per D-cache miss")
    g.add_argument("--bpred", default="gshare",
                   choices=["gshare", "bimodal", "always_not_taken"])
    g.add_argument("--data-dependent-div", action="store_true",
                   help="model an early-terminating iterative divider")
    g.add_argument("--noise", type=float, default=1.0,
                   help="sigma of additive Gaussian measurement noise")


def cmd_list(args) -> int:
    width = max(len(n) for n in victims.REGISTRY)
    for name, v in victims.REGISTRY.items():
        tag = "hardened" if v.hardened else "LEAKY   "
        print(f"{name:<{width}}  [{tag}]  {v.description}")
    return 0


def cmd_disasm(args) -> int:
    build = campaign._build(args.victim, bytes(16), campaign.DEFAULT_KEY)
    for line in build.program.disassemble():
        print(line)
    return 0


def cmd_tvla(args) -> int:
    cfg = _model_from_args(args)
    names = [args.victim] if args.victim != "all" else list(victims.REGISTRY)
    results = {}
    for name in names:
        res = campaign.tvla_campaign(name, args.traces, cfg=cfg,
                                     align_mode=args.align, seed=args.seed)
        results[name] = res.tvla
        print(res.summary())
        print(report.t_trace_ascii(res.tvla.t, res.tvla.threshold))
        print("-" * 78)
    if args.figures:
        for path in report.save_figures(args.figures, tvla_results=results):
            print(f"wrote {path}")
    return 0


def cmd_null(args) -> int:
    res = campaign.null_campaign(args.victim, args.traces,
                                 cfg=_model_from_args(args), seed=args.seed)
    print(res.summary())
    print(report.t_trace_ascii(res.tvla.t, res.tvla.threshold))
    if res.tvla.leaks:
        print("\nFALSE POSITIVE: the detector fired on a no-channel control.")
        return 1
    return 0


def cmd_cpa(args) -> int:
    cfg = _model_from_args(args)
    res = campaign.cpa_campaign(args.traces, target_byte=args.byte, cfg=cfg,
                                model=args.leakage_model, use_poi=not args.no_poi,
                                seed=args.seed)
    print(res.summary())
    print(report.correlation_ascii(res.result, res.true_key))
    if args.figures:
        for path in report.save_figures(args.figures, cpa_result=res.result,
                                        true_key=res.true_key):
            print(f"wrote {path}")
    return 0 if res.recovered else 1


def cmd_fullkey(args) -> int:
    res = campaign.full_key_campaign(args.traces, cfg=_model_from_args(args),
                                     model=args.leakage_model, seed=args.seed)
    print(res.summary())
    return 0 if res.n_correct == 16 else 1


def cmd_sweep(args) -> int:
    """Sweep measurement noise and report the cost of the attack at each level."""
    print(f"{'sigma':>8}  {'bytes ok':>8}  {'worst-case traces':>18}")
    for sigma in args.sigmas:
        cfg = _model_from_args(args)
        cfg.noise_sigma = sigma
        res = campaign.full_key_campaign(args.traces, cfg=cfg,
                                         model=args.leakage_model, seed=args.seed)
        worst = res.worst_case_traces
        print(f"{sigma:8.2f}  {res.n_correct:>6}/16  "
              f"{(worst if worst is not None else '> ' + str(args.traces)):>18}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvleak",
        description="Microarchitectural leakage analysis for RV32IM software.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="list available victim programs")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("disasm", help="disassemble a victim")
    sp.add_argument("victim", choices=list(victims.REGISTRY))
    sp.set_defaults(func=cmd_disasm)

    sp = sub.add_parser("tvla", help="fixed-vs-random leakage detection")
    sp.add_argument("victim", nargs="?", default="all",
                    choices=list(victims.REGISTRY) + ["all"])
    sp.add_argument("-n", "--traces", type=int, default=200)
    sp.add_argument("--align", default="pad", choices=["pad", "truncate"])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--figures", metavar="DIR", help="write PNG figures to DIR")
    _add_model_args(sp)
    sp.set_defaults(func=cmd_tvla)

    sp = sub.add_parser("null", help="fixed-vs-fixed false-positive control")
    sp.add_argument("victim", nargs="?", default="table-lookup",
                    choices=list(victims.REGISTRY))
    sp.add_argument("-n", "--traces", type=int, default=200)
    sp.add_argument("--seed", type=int, default=0)
    _add_model_args(sp)
    sp.set_defaults(func=cmd_null)

    sp = sub.add_parser("cpa", help="recover one key byte by correlation analysis")
    sp.add_argument("-n", "--traces", type=int, default=1000)
    sp.add_argument("-b", "--byte", type=int, default=0)
    sp.add_argument("--leakage-model", default="hd", choices=["hd", "hw"])
    sp.add_argument("--no-poi", action="store_true",
                    help="disable attribution-guided point-of-interest selection")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--figures", metavar="DIR")
    _add_model_args(sp)
    sp.set_defaults(func=cmd_cpa)

    sp = sub.add_parser("fullkey", help="recover all 16 key bytes")
    sp.add_argument("-n", "--traces", type=int, default=1000)
    sp.add_argument("--leakage-model", default="hd", choices=["hd", "hw"])
    sp.add_argument("--seed", type=int, default=7)
    _add_model_args(sp)
    sp.set_defaults(func=cmd_fullkey)

    sp = sub.add_parser("sweep", help="attack cost as a function of noise")
    sp.add_argument("-n", "--traces", type=int, default=1000)
    sp.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0, 8.0])
    sp.add_argument("--leakage-model", default="hd", choices=["hd", "hw"])
    sp.add_argument("--seed", type=int, default=7)
    _add_model_args(sp)
    sp.set_defaults(func=cmd_sweep)

    return p


def main(argv=None) -> int:
    # Restore default SIGPIPE handling so that piping output into head, less, or
    # similar terminates quietly instead of raising BrokenPipeError. Python
    # installs SIG_IGN by default, which turns a normal shell idiom into a
    # traceback.
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass  # not available on Windows or in a non-main thread
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
