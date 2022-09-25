import torch

from .layers import EmbeddingLayer, MultiLayerPerceptron


class DLRMModel(torch.nn.Module):
    """
    A pytorch implementation of DLRM Model.

    Reference:
        Maxim Naumov, et al. Deep Learning Recommendation Model for Personalization and Recommendation Systems.
        https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, up_mlp_dims, dropout):
        super().__init__()
        print(
            "categorical_field_dims = {}, numerical_num = {}, embed_dim = {}, bottom_mlp_dims = {}, up_mlp_dims = {}, dropout = {}".format(
                categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, up_mlp_dims, dropout
            )
        )
        # len(categorical_field_dims) = 24, numerical_num = 2, embed_dim = 128, bottom_mlp_dims = (32, 16), up_mlp_dims = (256, 128, 64), dropout = 0.2
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.catgorical_out_dim = len(categorical_field_dims) ** 2  # 24*24 = 576
        self.interaction_output_dim = self.catgorical_out_dim + bottom_mlp_dims[-1]  # 576 + 16 = 64

        self.bottom_mlp = MultiLayerPerceptron(numerical_num, bottom_mlp_dims, dropout, output_layer=False)
        self.top_mlp = MultiLayerPerceptron(self.interaction_output_dim, up_mlp_dims, dropout, output_layer=True, need_sigmoid=True)

    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)`` # [2048, 24])
        numerical_x: Long tensor of size ``(batch_size, numerical_num)`` # [2048, 2]
        """

        categorical_emb = self.embedding(categorical_x)  # [2048, 24, 128]
        numerical_out = self.bottom_mlp(numerical_x)  # [2048, 16]
        categorical_out = torch.bmm(categorical_emb, torch.transpose(categorical_emb, 1, 2))  # [2048, 24, 24]
        categorical_out = categorical_out.view(-1, self.catgorical_out_dim)  # [2048, 576]
        interaction_out = torch.cat([numerical_out, categorical_out], 1).view(
            -1, self.interaction_output_dim
        )  # [2048, 592]
        results = [self.top_mlp(interaction_out)]  # [2048, 1]
        return results
