from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_kl_loss(parent_logits: torch.Tensor, child_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    parent_log = F.log_softmax(parent_logits.float() / temperature, dim=-1)
    child_log = F.log_softmax(child_logits.float() / temperature, dim=-1)
    return (temperature**2) * F.kl_div(child_log, parent_log.exp(), reduction="batchmean")


def anchor_loss(current: torch.Tensor, initial: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((current.float() - initial.float()) * mask.float()) ** 2).mean()


def activation_l2(parent_hidden: torch.Tensor, child_hidden: torch.Tensor) -> torch.Tensor:
    return ((parent_hidden.float() - child_hidden.float()) ** 2).mean()


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
