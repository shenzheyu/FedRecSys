import torch
import torch.nn as nn
import numpy as np


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.need_sigmoid = output_size > 0
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class MMOE(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden, towers_hidden):
        super(MMOE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for i in range(2)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden), Tower(self.experts_out, 1, self.towers_hidden)])

        self.layer_norm = nn.LayerNorm(experts_out)

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [self.layer_norm(torch.sum(ti, dim=0)) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return final_output


class SparseMMOE(nn.Module):
    def __init__(self, embedding_list, pretrained_embeddings=None, embedding2idxes=None):
        super(SparseMMOE, self).__init__()
        self.MMOE = MMOE(input_size=73, num_experts=6, experts_out=32, experts_hidden=64, towers_hidden=8)
        self.embeddings = nn.ModuleDict()
        if pretrained_embeddings is not None:
            for embedding_key in embedding_list:
                self.embeddings[embedding_key] = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(pretrained_embeddings[embedding_key]), freeze=False)
        if embedding2idxes is not None:
            self.embedding2idxes = embedding2idxes
        self.batch_norms = nn.ModuleDict()
        self.batch_norms['movie_year'] = nn.BatchNorm1d(1)
        self.batch_norms['rating_timestamp'] = nn.BatchNorm1d(1)
        # self.glove_embedding = load_glove()
        # self.movie_title_lstm = nn.LSTM()

    def forward(self, features_list):
        # sparse feature
        user_id_input = self.embeddings['user_id'](torch.LongTensor([self.embedding2idxes['user_id'][features['user_id']] for features in features_list]))
        user_zipcode_input = self.embeddings['user_zipcode'](torch.LongTensor([self.embedding2idxes['user_zipcode'][features['user_zipcode']] for features in features_list]))
        movie_id_input = self.embeddings['movie_id'](torch.LongTensor([self.embedding2idxes['movie_id'][features['movie_id']] for features in features_list]))

        # dense feature
        user_gender_input = torch.FloatTensor([[features['user_gender']] for features in features_list])
        user_age_input = torch.FloatTensor([features['user_age'] for features in features_list])
        user_occupation_input = torch.FloatTensor([features['user_occupation'] for features in features_list])
        movie_year_input = torch.FloatTensor([[features['movie_year']] for features in features_list])
        movie_year_input = self.batch_norms['movie_year'](movie_year_input)
        movie_genres_input = torch.FloatTensor([features['movie_genres'] for features in features_list])
        rating_timestamp_input = torch.FloatTensor([[features['rating_timestamp']] for features in features_list])
        rating_timestamp_input = self.batch_norms['rating_timestamp'](rating_timestamp_input)

        # sequence feature
        # movie_title_input = self.movie_title_lstm(features['movie_title'])

        mmoe_input = torch.cat((user_id_input, user_zipcode_input, movie_id_input, user_gender_input, user_age_input,
                                user_occupation_input, movie_year_input, movie_genres_input, rating_timestamp_input), 1)

        output = self.MMOE(mmoe_input)
        return output

    def load_glove(self):
        words = []
        idx = 0
        word2idx = {}
        vectors = []

        with open('data/glove.6B.100d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
