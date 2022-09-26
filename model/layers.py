import torch
import numpy as np

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.embedding_map = None
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def update_embedding_map(self, embedding_map):
        self.embedding_map = embedding_map

    def update_embedding(self, embedding_weight):
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight, freeze=False)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        if self.embedding_map is not None:
            x.map_(x, lambda i, *y: self.embedding_map[i])
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, need_sigmoid=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
            layers.append(torch.nn.Sigmoid())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)