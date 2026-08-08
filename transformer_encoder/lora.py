"""LoRA adapters and utilities for freezing the base encoder."""
import math
from torch import nn


class LoRALinear(nn.Module):
    """A frozen base projection plus trainable low-rank update: W + (B @ A) * scale."""
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank, self.scale = rank, alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def apply_lora(model: nn.Module, rank: int = 4, alpha: float = 8.0):
    """Freeze all model parameters, retain classifier, and adapt attention Q/V projections."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.blocks:
        block.attention.q_proj = LoRALinear(block.attention.q_proj, rank, alpha)
        block.attention.v_proj = LoRALinear(block.attention.v_proj, rank, alpha)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
