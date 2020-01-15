# https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/gmf.py

import torch

from model.embedding import EmbeddingGrad


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_item = EmbeddingGrad(num_embedding=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        # self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        item_embedding = self.embedding_item(item_indices)
        user_embedding = self.embedding_user(user_indices)

        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        # rating = self.logistic(logits)
        return logits

    def get_input_grad(self, indices):
        """
        Get gradients with respect to inputs
        :param indices: (ndarray) array of item indices
        :return: (tensor) tensor of gradients with respect to inputs
        """
        if indices.ndim == 1:
            indices = indices.reshape(-1, 1)

        dims = [d for d in indices.shape] + [1]
        idx_tensor = torch.LongTensor(indices).reshape(dims)

        grad = self.embedding_item.get_grad(indices)
        grad_at_idx = torch.gather(grad, -1, idx_tensor)
        return torch.squeeze(grad_at_idx)

    def init_weight(self):
        pass
