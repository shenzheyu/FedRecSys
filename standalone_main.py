import os

import numpy as np
import torch
import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.aliexpress import AliExpressDataset
from dataset.movielens import MovieLensDataset, MovieLensDatasetWithBehavior
from model.din import DINModel
from model.dlrm import DLRMModel
from model.lr import LinearRegression
from model.mmoe import MMoEModel
from model.ple import PLEModel
from model.wdl import WDLModel
from util.options import args_parser
from util.utils import count_parameters


def get_dataset(name, path, model_name, task_num):
    if 'AliExpress' in name:
        return AliExpressDataset(path, task_num=task_num)
    if 'MovieLens' in name:
        if model_name == 'din':
            return MovieLensDatasetWithBehavior(path, task_num=task_num)
        else:
            return MovieLensDataset(path, task_num=task_num)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'mmoe':
        print('Model: MMoE')
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print('Model: PLE')
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                        tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
                        specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'din':
        print('Model: DIN')
        model = DINModel(categorical_field_dims, numerical_num, embed_dim=embed_dim)
        model.embedding.update_offsets(None)
        return model
    elif name == 'dlrm':
        print('Model: DLRM')
        return DLRMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(32, 16),
                         up_mlp_dims=(256, 128, 64), dropout=0.2)
    elif name == 'wdl':
        print('Model: Wide&Deep')
        return WDLModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, deep_mlp_dims=(1024, 512, 256),
                        dropout=0.2)
    elif name == 'lr':
        print('Model: LinearRegression')
        return LinearRegression(categorical_field_dims, numerical_num)
    else:
        raise ValueError('unknown model name: ' + name)


def get_criterion(name):
    criterion = []
    for n in name.split(','):
        if n == 'bce':
            criterion.append(torch.nn.BCELoss())
        elif n == 'mse':
            criterion.append(torch.nn.MSELoss(reduction='mean'))
        else:
            raise ValueError('unknown criterion name: ' + n)
    return criterion


def get_evaluation(name):
    evaluation = []
    for n in name.split(','):
        if n == 'auc':
            evaluation.append(roc_auc_score)
        elif n == 'rmse':
            evaluation.append(lambda y, y_hat: mean_squared_error(y, y_hat, squared=False))
        else:
            raise ValueError('unknown evaluation name: ' + n)
    return evaluation


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(args, model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
            device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        loss_list = [criterion[i](y[i].view(-1), labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

        # Export the model
        if not args.is_onnx_exported:
            torch.onnx.export(model,  # model being run
                              (categorical_fields, numerical_fields),  # model input (or a tuple for multiple inputs)
                              args.model_name + ".onnx",  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=10,  # the ONNX version to export the model to
                              do_constant_folding=False,  # whether to execute constant folding for optimization
                              input_names=['input'],  # the model's input names
                              output_names=['output'],  # the model's output names
                              )
            args.is_onnx_exported = True


def train_din(args, model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (categorical_fields, numerical_fields, query_item, user_behavior, user_behavior_length, labels) in enumerate(
            loader):
        categorical_fields, numerical_fields, query_item, user_behavior, user_behavior_length, labels = categorical_fields.to(
            device), numerical_fields.to(device), query_item.to(device), user_behavior.to(
            device), user_behavior_length.to(device), labels.to(device)
        y = model(categorical_fields, numerical_fields, query_item, user_behavior, user_behavior_length)
        loss_list = [criterion[i](y[i].view(-1), labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, task_num, criterion, evaluation, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].append(criterion[i](y[i].view(-1).cpu(), labels[:, i].float().cpu()))
    evaluation_results, loss_results = list(), list()
    for i in range(task_num):
        evaluation_results.append(evaluation[i](labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return evaluation_results, loss_results


def test_din(model, data_loader, task_num, criterion, evaluation, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, query_item, user_behavior, user_behavior_length, labels in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, query_item, user_behavior, user_behavior_length, labels = categorical_fields.to(
                device), numerical_fields.to(device), query_item.to(device), user_behavior.to(
                device), user_behavior_length.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields, query_item, user_behavior, user_behavior_length)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].append(criterion[i](y[i].view(-1).cpu(), labels[:, i].float().cpu()))
    evaluation_results, loss_results = list(), list()
    for i in range(task_num):
        evaluation_results.append(evaluation[i](labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return evaluation_results, loss_results


def main(args, dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         criterion_name,
         evaluation_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         use_early_stopper,
         device,
         save_dir):
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv', model_name,
                                args.task_num)
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv', model_name,
                               args.task_num)
    if model_name == 'din':
        train_dataset.get_user_behaviors()
        train_dataset.update_behavior_data()
        train_dataset.add_offset()
        test_dataset.get_user_behaviors(train_dataset.user_behavior_dict)
        test_dataset.update_behavior_data()
        test_dataset.add_offset()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    model = get_model(model_name, field_dims, numerical_num, task_num, expert_num, embed_dim).to(device)
    print(model)
    print("model_size = {}".format(count_parameters(model)))  # DLRM - 1.4M

    criterion = get_criterion(criterion_name)
    evaluation_func = get_evaluation(evaluation_name)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_path = f'{save_dir}/{dataset_name}_{model_name}.pt'
    early_stopper = EarlyStopper(num_trials=2, save_path=save_path)
    for epoch_i in range(epoch):
        if model_name == 'din':
            train_din(args, model, optimizer, train_data_loader, criterion, device)
            evaluation, loss = test_din(model, test_data_loader, task_num, criterion, evaluation_func, device)
        else:
            train(args, model, optimizer, train_data_loader, criterion, device)
            evaluation, loss = test(model, test_data_loader, task_num, criterion, evaluation_func, device)
        print('epoch:', epoch_i)
        for i in range(task_num):
            print(f'task {i}, {evaluation_name.split(",")[i]} {evaluation[i]}, Log-loss {loss[i]}')
        if use_early_stopper and not early_stopper.is_continuable(model, np.array(evaluation).mean()):
            print(f'test: best evaluation: {early_stopper.best_accuracy}')
            break

    if use_early_stopper:
        model.load_state_dict(torch.load(save_path))
    if model_name == 'din':
        evaluation, loss = test_din(model, test_data_loader, task_num, criterion, evaluation_func, device)
    else:
        evaluation, loss = test(model, test_data_loader, task_num, criterion, evaluation_func, device)
    f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print(f'task {i}, {evaluation_name.split(",")[i]} {evaluation[i]}, Log-loss {loss[i]}')
        f.write(f'task {i}, {evaluation_name.split(",")[i]} {evaluation[i]}, Log-loss {loss[i]}')
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    args = args_parser()
    main(args, args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.criterion_name,
         args.evaluation_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.use_early_stopper,
         args.device,
         args.save_dir)
