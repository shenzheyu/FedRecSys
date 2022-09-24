import torch

from .layers import EmbeddingLayer, MultiLayerPerceptron


class DLRMModel(torch.nn.Module):
    """
    A pytorch implementation of DLRM Model.

    Reference:
        Maxim Naumov, et al. Deep Learning Recommendation Model for Personalization and Recommendation Systems.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, up_mlp_dims, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.catgorical_out_dim = len(categorical_field_dims) ** 2
        self.interaction_output_dim = self.catgorical_out_dim + bottom_mlp_dims[-1]

        self.bottom_mlp = MultiLayerPerceptron(numerical_num, bottom_mlp_dims, dropout, output_layer=False)
        self.top_mlp = MultiLayerPerceptron(self.interaction_output_dim, up_mlp_dims, dropout, output_layer=True)

    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_out = self.bottom_mlp(numerical_x)
        categorical_out = torch.bmm(categorical_emb, torch.transpose(categorical_emb, 1, 2)).view(-1, self.catgorical_out_dim)
        interaction_out = torch.cat([numerical_out, categorical_out], 1).view(-1, self.interaction_output_dim)
        results = [self.top_mlp(interaction_out)]
        return results
