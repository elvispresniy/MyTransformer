from attention import Transformer
import torch
import torch.nn as nn
from torch import Tensor, BoolTensor

class TransformerForLM(Transformer):
    def __init__(self, loss_fn: nn.Module = nn.CrossEntropyLoss, **kwargs):
        super().__init__(**kwargs)

        self.loss_fn = loss_fn

    def forward_with_loss(self, x: Tensor, mask: BoolTensor|None = None):
        input = x[:, :-1]
        labels = x[:, 1:]

        logits = super().forward(input, mask)

        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return {"logits": logits, "loss": loss}
    
