from __future__ import annotations

import argparse
from pathlib import Path

from kt_kernel.experts_base import BaseMoEWrapper

from .trace_collector import RoutingTraceCollector


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("moe-routing-collect")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-id", required=True)
    p.add_argument("--context-id", required=True)
    p.add_argument("--token-category", default="assistant")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    collector = RoutingTraceCollector(args.output_dir, args.prompt_id, args.token_category)
    collector.start(context_id=args.context_id)

    def hook(layer_id, topk_ids, topk_weights, all_expert_scores=None):
        per_token_scores = None
        if all_expert_scores is not None:
            per_token_scores = all_expert_scores.to(dtype=all_expert_scores.dtype, device="cpu")
        for pos in range(topk_ids.shape[0]):
            collector.record(
                layer_id=layer_id,
                token_position=pos,
                expert_ids=topk_ids[pos].tolist(),
                expert_weights=topk_weights[pos].tolist(),
                expert_scores_all=(per_token_scores[pos].tolist() if per_token_scores is not None else None),
            )

    BaseMoEWrapper.set_trace_hook(hook)
    # Inference run is external to this command's minimal harness contract.


if __name__ == "__main__":
    main()
