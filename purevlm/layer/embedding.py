import torch

class Embedding:
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = None

    def __call__(self, input):
        return torch.nn.functional.embedding(input, self.weight, self.padding_idx, None, 2.0, False, False)