import torch

from kt_kernel.experts_base import BaseMoEWrapper


def test_trace_hook_invoked_for_submit_forward(monkeypatch):
    calls = []

    def hook(layer_id, topk_ids, topk_weights):
        calls.append((layer_id, topk_ids.shape, topk_weights.shape))

    BaseMoEWrapper.set_trace_hook(hook)
    hook_fn = BaseMoEWrapper.get_trace_hook()
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    w = torch.tensor([[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]], dtype=torch.float32)
    hook_fn(3, x, w)
    assert calls[0][0] == 3
    BaseMoEWrapper.set_trace_hook(None)


def test_trace_hook_accepts_optional_full_scores():
    calls = []

    def hook(layer_id, topk_ids, topk_weights, all_expert_scores=None):
        calls.append((layer_id, all_expert_scores.shape if all_expert_scores is not None else None))

    BaseMoEWrapper.set_trace_hook(hook)
    hook_fn = BaseMoEWrapper.get_trace_hook()
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    w = torch.tensor([[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]], dtype=torch.float32)
    s = torch.tensor([[0.1, 0.3, 0.6]], dtype=torch.float16)
    hook_fn(3, x, w, s)
    assert calls[0][0] == 3
    assert calls[0][1] == (1, 3)
    BaseMoEWrapper.set_trace_hook(None)
