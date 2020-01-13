import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.embedding import EmbeddingGrad
from preprocessing.utils import split_train_test_user, load_dict_output
import pandas as pd
from generator.generator import CoocurrenceGenerator, SimpleBatchGenerator
from model._loss import loss_mse, utility_loss, mrs_loss




class NeuralUtility(nn.Module):

    def __init__(self, backbone, n_items, h_dim_size):
        super(NeuralUtility, self).__init__()

        self.embedding = EmbeddingGrad(n_items, h_dim_size)
        self.backbone = backbone


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

        grad = self.embedding.get_grad(indices)
        grad_at_idx = torch.gather(grad, -1, idx_tensor)
        return torch.squeeze(grad_at_idx)

    def forward(self, x):

        e_i = self.embedding.forward(x)
        y_hat = self.backbone.forward(e_i)

        return y_hat


    def predict(self, X_test):
        return self.forward(X_test)


