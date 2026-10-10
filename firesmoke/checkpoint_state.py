"""Full checkpoint restoration, distinct from pretrained initialization."""
from __future__ import annotations

import torch


def restore_full_model_state(model, source):
    source_state = source.float().state_dict()
    model.load_state_dict(source_state, strict=True)
    restored = model.state_dict()
    mismatched = [name for name, tensor in source_state.items()
                  if not torch.equal(restored[name].detach().cpu(), tensor.detach().cpu())]
    if mismatched:
        raise RuntimeError("Full resume state mismatch: " + ", ".join(mismatched[:8]))
    return {"policy": "strict_full_checkpoint_state", "passed": True,
            "tensors": len(source_state),
            "head_tensors": sum(name.startswith("model.23.") for name in source_state),
            "refinement_tensors": sum(name.startswith("model.23.refinement.") for name in source_state),
            "elements": sum(tensor.numel() for tensor in source_state.values()),
            "mismatched_tensors": 0}
