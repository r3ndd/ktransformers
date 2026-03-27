from pathlib import Path

from kt_kernel.moe_routing.collect import build_arg_parser


def test_collect_parser_defaults(tmp_path: Path):
    p = build_arg_parser()
    args = p.parse_args(["--output-dir", str(tmp_path), "--prompt-id", "p1", "--context-id", "c1"])
    assert args.prompt_id == "p1"
