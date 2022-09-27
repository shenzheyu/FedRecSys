import torch

from .layers import EmbeddingLayer, MultiLayerPerceptron


class WDLModel(torch.nn.Module):
    """
    A pytorch implementation of Wide&Deep Learning Model.

    Reference:
        Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]
        //Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, deep_mlp_dims, dropout):
        super().__init__()
        print(
            "categorical_field_dims = {}, numerical_num = {}, embed_dim = {}, deep_mlp_dims = {}, dropout = {}".format(
                categorical_field_dims, numerical_num, embed_dim, deep_mlp_dims, dropout
            )
        )
        # len(categorical_field_dims) = 24, numerical_num = 2, embed_dim = 128, deep_mlp_dims = (1024, 512, 256), dropout = 0.2
        self.categorical_field_dims = categorical_field_dims
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.embed_output_dim = len(categorical_field_dims) * embed_dim
        self.wide_mlp = torch.nn.Linear(numerical_num + len(categorical_field_dims), 1)
        self.deep_mlp = torch.nn.Sequential(MultiLayerPerceptron(self.embed_output_dim, deep_mlp_dims, dropout, output_layer=False), torch.nn.Linear(deep_mlp_dims[-1], 1))
        self.out_unit = torch.nn.Sigmoid()

    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)`` # [2048, 24])
        numerical_x: Long tensor of size ``(batch_size, numerical_num)`` # [2048, 2]
        """

        wide_out = self.wide_mlp(torch.cat([(categorical_x / self.categorical_field_dims).float(), numerical_x], 1)) # [2048, 1]
        deep_emb = self.embedding(categorical_x).view(-1, self.embed_output_dim) # [2048, 3027]
        deep_out = self.deep_mlp(deep_emb) # [2048, 1]
        logit = wide_out + deep_out
        return [self.out_unit(logit)]
