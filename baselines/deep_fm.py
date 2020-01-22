# source: https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/dfm.py

import torch

from baselines.fm_layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
from baselines.fm_layers import FeaturesSparseEmbedding, FeaturesSparseLinear


class DeepFM(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_logit=False):
        super().__init__()
        self.use_logit = use_logit
        self.user_linear = FeaturesSparseLinear(field_dims[0])
        self.item_linear = FeaturesLinear(field_dims[1])

        self.fm = FactorizationMachine(reduce_sum=False)

        self.user_embedding = FeaturesSparseEmbedding(field_dims[0], embed_dim)
        self.item_embedding = FeaturesEmbedding(field_dims[1], embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, users, items):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        user_embed = self.user_embedding(users).squeeze()
        item_embed = self.item_embedding(items)

        user_linear = self.user_linear(users)
        item_linear = self.item_linear(items).unsqueeze(dim=-1)

        if item_embed.ndim == 3:
            user_embed = user_embed.unsqueeze(1).repeat(1, items.shape[1], 1)
            user_linear = user_linear.unsqueeze(1).repeat(1, items.shape[1], 1)


        x_embed = torch.cat([user_embed, item_embed], dim=-1)

        x_fm = self.fm(x_embed).unsqueeze(dim=-1)
        x_mlp = self.mlp(x_embed.view(-1, self.embed_output_dim))

        if item_embed.ndim == 3:
            x_mlp = x_mlp.view(-1, items.shape[1], 1)

        output = user_linear + item_linear + x_fm + x_mlp

        if self.use_logit:
            output = torch.sigmoid(output)

        return output