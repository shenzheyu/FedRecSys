import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

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
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class MMOE(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden, towers_hidden, tasks):
        super(MMOE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return final_output


class SparseMMOE(nn.Module):
    def __init__(self, feature_num, pretrained_embedding, embedding_map, num_experts, experts_out,
                 experts_hidden, towers_hidden, tasks):
        super(SparseMMOE, self).__init__()
        input_size = len(pretrained_embedding[0]) * feature_num
        self.MMOE = MMOE(input_size, num_experts, experts_out, experts_hidden, towers_hidden, tasks)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embedding))
        self.embedding.requires_grad_(requires_grad=True)
        self.embedding_map = embedding_map

    def forward(self, x):
        embedding_input = torch.tensor([[self.embedding_map[feature.item()] for feature in features] for features in x])
        embedding_output = self.embedding(embedding_input).view([x.shape[0], -1])
        output = self.MMOE(embedding_output)
        return output
