import torch
from .layers import EmbeddingLayer


class Dice(torch.nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = torch.nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = torch.nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = torch.zeros((num_features, 1))
        elif self.dim == 2:
            self.alpha = torch.zeros((num_features,))

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x

        return out


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True, dropout_rate=0.5, activation='relu',
                 sigmoid=False, dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = [torch.nn.Linear(input_size, hidden_size[0], bias=bias[0])]

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(torch.nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(torch.nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(torch.nn.Dropout(p=dropout_rate))
            layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias[i]))

        self.fc = torch.nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = torch.nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class AttentionSequencePoolingLayer(torch.nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_size=[64, 16], bias=[True, True], embedding_dim=embedding_dim,
                                             batch_norm=False)

    def forward(self, query_ad, user_behavior, user_behavior_length):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # user behavior length: size -> batch_size * 1
        # output              : size -> batch_size * 1 * embedding_size

        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        # print(attention_score.size())

        # define mask by length
        user_behavior_length = user_behavior_length.type(torch.LongTensor)
        mask = torch.arange(user_behavior.size(1))[None, :] < user_behavior_length[:, None]

        # mask
        output = torch.mul(attention_score, mask.type(torch.FloatTensor))  # batch_size *

        # multiply weight
        output = torch.matmul(output, user_behavior)

        return output


class LocalActivationUnit(torch.nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4 * embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)
        # TODO: fc_2 initialization

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output


class DINModel(torch.nn.Module):
    """
    A pytorch implementation of DIN Model.

    Reference:
        [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim):
        super().__init__()
        print(
            "categorical_field_dims = {}, numerical_num = {}, embed_dim = {}".format(
                categorical_field_dims, numerical_num, embed_dim
            )
        )
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.attention = AttentionSequencePoolingLayer(embedding_dim=embed_dim)
        self.fc_layer = FullyConnectedLayer(input_size=embed_dim + len(categorical_field_dims) * embed_dim + numerical_num,
                                            hidden_size=[200, 80, 1],
                                            bias=[True, True, False],
                                            activation='relu',
                                            sigmoid=True)

    def forward(self, categorical_x, numerical_x, query_item, user_behavior, user_behavior_length):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)`` # [2048, 24])
        numerical_x: Float tensor of size ``(batch_size, numerical_num)`` # [2048, 2]
        query_item: Long tensor of size ``(batch_size)``
        user_behavior: Long tensor of size ``(batch_size, seq_num)``
        user_behavior_length: Long tensor of size ``(batch_size)``
        """
        query_item_emb = self.embedding(query_item)
        user_behavior_emb = self.embedding(user_behavior)
        history = self.attention(torch.unsqueeze(query_item_emb, 1), user_behavior_emb, torch.unsqueeze(user_behavior_length, 1))
        categorical_emb = self.embedding(categorical_x)  # [batch, categorical_field, embedding_size]
        concat_feature = torch.cat([categorical_emb.view(categorical_emb.shape[0], -1), numerical_x, history.squeeze()], dim=1)
        out = [self.fc_layer(concat_feature)]
        return out
