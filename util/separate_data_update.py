import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.aliexpress import AliExpressDataset
from dataset.movielens import MovieLensDataset
from util.update import LocalUpdate
from util.utils import get_criterion
from util.utils import get_evaluation


class SeparatedDataLocalUpdate(LocalUpdate):
    def __init__(self, args, data_path, field_dims, global_model, logger):
        self.args = args
        self.logger = logger
        self.field_dims = field_dims
        self.dataset = self.get_dataset(data_path)
        self.embedding_map = self.get_embedding_map()
        self.data2local(self.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.local_bs, shuffle=True)
        self.update_times = self.get_update_times()
        self.device = torch.device(args.device)
        self.criterion = get_criterion(self.args.criterion_name, self.device)
        self.evaluation_func = get_evaluation(self.args.evaluation_name)
        self.model = copy.deepcopy(global_model)
        self.model = self.model.to(self.device)
        self.model.embedding.update_offsets(None)
        # change the embedding tensor to personalized shape according to self.embedding_map
        self.fetch_embedding(global_model, True)

    def get_dataset(self, dataset_path):
        if "AliExpress" in self.args.dataset_name:
            dataset = AliExpressDataset(dataset_path, 0)
        elif "MovieLens" in self.args.dataset_name:
            dataset = MovieLensDataset(dataset_path, 0)
        else:
            raise ValueError("unknown dataset name: " + self.args.dataset_name)
        return dataset

    def data2local(self, dataset, **kwargs):
        global_offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        for sample_idx, sample in enumerate(dataset):
            map_function = lambda x: self.embedding_map[x]
            dataset.update_categorical_data(sample_idx, np.vectorize(map_function)(sample[0] + global_offsets))

    def get_embedding_map(self, **kwargs):
        embedding_map = {}
        global_offset = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        local_embedding_word = 0
        for categorical_fields, _, _ in self.dataset:
            for i in range(len(self.field_dims)):
                if categorical_fields[i] + global_offset[i] not in embedding_map.keys():
                    embedding_map[categorical_fields[i] + global_offset[i]] = local_embedding_word
                    local_embedding_word += 1
        return embedding_map  # (global_embedding_word, local_embedding_word)

    def get_update_times(self, **kwargs):
        update_times = {}  # {local_word -> update_times}
        for categorical_fields, _, _ in self.dataset:
            for i in range(len(self.field_dims)):
                if categorical_fields[i] in update_times.keys():
                    update_times[categorical_fields[i]] += 1
                else:
                    update_times[categorical_fields[i]] = 1
        return update_times

    def update_weights(self):
        # Set mode to train model
        self.model.train()
        self.model = self.model.to(self.device)
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(self.dataloader):
                # skip the one-size batch to avoid error in batch_norm
                if categorical_fields.size(0) <= 1:
                    continue

                categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(
                    self.device), labels.to(self.device)

                y = self.model(categorical_fields, numerical_fields)
                loss_list = [self.criterion[i](y[i].view(-1), labels[:, i].float()) for i in range(labels.size(1))]
                loss = 0
                for item in loss_list:
                    loss += item
                loss /= len(loss_list)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) != 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self):
        """ Returns the inference accuracy and loss.
        """

        self.model.eval()
        self.model = self.model.to(self.device)
        labels_dict, predicts_dict, loss_dict = {}, {}, {}
        for i in range(self.args.task_num):
            labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()

        with torch.no_grad():
            for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(self.dataloader):
                categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(
                    self.device), labels.to(self.device)

                # Inference
                y = self.model(categorical_fields, numerical_fields)
                for i in range(self.args.task_num):
                    if y[i].size(0) <= 1:
                        labels_dict[i].append(labels[:, i].tolist())
                        predicts_dict[i].append(y[i].tolist())
                        loss_dict[i].append(
                            self.criterion[i](y[i].view(-1), labels[:, i].float()).tolist())
                    else:
                        labels_dict[i].extend(labels[:, i].tolist())
                        predicts_dict[i].extend(y[i].tolist())
                        loss_dict[i].append(
                            self.criterion[i](y[i].view(-1), labels[:, i].float()).tolist())

        evaluation_results, loss_results = list(), list()
        for i in range(self.args.task_num):
            evaluation_results.append(self.evaluation_func[i](labels_dict[i], predicts_dict[i]))
            loss_results.append(np.array(loss_dict[i]).mean())
        return evaluation_results, loss_results
