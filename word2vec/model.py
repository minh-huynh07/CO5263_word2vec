
import torch
from torch import nn

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center: torch.Tensor, context: torch.Tensor = None, mode: str = 'dot') -> torch.Tensor:
        v_c = self.input_embeddings(center)  # (batch_size, embed_dim)
        if mode == 'softmax':
            # Return logits for entire vocabulary
            logits = torch.matmul(v_c, self.output_embeddings.weight.t())  # (batch_size, vocab_size)
            return logits
        else:
            # context: (batch_size,)
            u_o = self.output_embeddings(context)     # (batch_size, embed_dim)
            score = (v_c * u_o).sum(dim=1)            # (batch_size,)
            return score

    def get_input_embedding(self) -> torch.Tensor:
        return self.input_embeddings.weight.data

    def get_output_embedding(self) -> torch.Tensor:
        return self.output_embeddings.weight.data
