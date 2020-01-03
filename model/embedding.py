import torch.nn as nn
import torch
import numpy as np

class EmbeddingGrad(nn.Module):

    def __init__(self, num_embedding, embedding_dim, init_embed=None):
        super(EmbeddingGrad, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.input = torch.eye(num_embedding, requires_grad=True)
        self.weights = nn.Linear(num_embedding, embedding_dim)

        if init_embed is not None:
            self.weights.weight = torch.nn.Parameter(init_embed)


    def init_embed(self):
        pass

    def _collect(self, indices):
        if indices.ndim == 1:
            x = self.input[indices]
        elif indices.ndim == 2:
            dims = indices.shape
            x = torch.zeros((dims[0], dims[1], self.num_embedding))
            for i, row in enumerate(indices):
                x[i] = self.input[row]
        return x


    def forward(self, indices):
        x = self._collect(indices)
        e = self.weights(x)
        return e

    def get_grad(self, indices):

        dims = [d for d in indices.shape] + [self.num_embedding]
        grad = torch.zeros(dims)
        for i, row in enumerate(indices):
            grad[i] = self.input.grad[row]

        return grad











if __name__ == "__main__":

    init_embed = np.ones((5,8))

    init_tensor = torch.from_numpy(init_embed)
    E = EmbeddingGrad(5, 8, init_embed=init_tensor)
    idx = np.array([2, 2, 0])
    e = E(idx)

    e_bar = e.mean()
    e_bar.backward()
    print("mean: ", e_bar)
    g = E.get_grad(idx)
    print(g)