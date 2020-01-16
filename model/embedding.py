import numpy as np
import torch
import torch.nn as nn


class EmbeddingGrad(nn.Module):

    def __init__(self, num_embedding, embedding_dim, init_embed=None, use_cuda=False):
        super(EmbeddingGrad, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.input = torch.eye(num_embedding, requires_grad=True).to(self.device)
        self.weights = nn.Linear(num_embedding, embedding_dim).cuda()

        if init_embed is not None:
            self.weights.weight = torch.nn.Parameter(init_embed)

        if use_cuda:
            self = self.cuda()

    def init_embed(self):
        pass

    def _collect(self, indices):
        if indices.ndim == 1:
            x = self.input[indices]
        else:
            dims = list(indices.shape)
            #x = torch.zeros((dims[0], dims[1], self.num_embedding))
            x = torch.zeros(dims + [self.num_embedding])
            for i, row in enumerate(indices):
                x[i] = self.input[row]

        return x.to(self.device)

    def forward(self, indices):
        x = self._collect(indices)
        e = self.weights(x)
        return e

    def get_grad(self, indices):

        dims = [d for d in indices.shape] + [self.num_embedding]
        grad = torch.zeros(dims)
        for i, row in enumerate(indices):
            grad[i] = self.input.grad[row]

        return grad.to(self.device)


if __name__ == "__main__":
    init_embed = np.ones((5, 8))

    init_tensor = torch.from_numpy(init_embed)
    E = EmbeddingGrad(5, 8, init_embed=init_tensor)
    idx = np.array([2, 2, 0])
    e = E(idx)

    e_bar = e.mean()
    e_bar.backward()
    print("mean: ", e_bar)
    g = E.get_grad(idx)
    print(g)
