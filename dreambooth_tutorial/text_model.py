import torch
import torch.nn as nn

# TextModel class which has a parameterized embedding
# in the forward pass we return that embedding


class NCModel(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        hidden_size,
        init_embedding=None,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(
            torch.randn(
                4, 64, 64
            ) / 10000
            if init_embedding is None
            else init_embedding
        )

    def forward(self, bs):
        # make embedding batch size specific
        return torch.tanh(self.embedding.repeat(bs, 1, 1, 1))
