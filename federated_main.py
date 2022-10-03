import copy
import os
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from util.options import args_parser
from util.separate_data_update import SeparatedDataLocalUpdate
from util.update import DatasetSplit, LocalUpdate, test_inference
from util.utils import get_model, get_dataset, average_weights, average_embeddings

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    # exp_details(args)

    device = torch.device(args.device)

    # load dataset and user groups
    if args.separate_data:
        dataset, _ = get_dataset(args.dataset_name,
                                 os.path.join(args.dataset_path, args.dataset_name) + '/test.csv',
                                 0)
    else:
        dataset, user_groups = get_dataset(args.dataset_name,
                                           os.path.join(args.dataset_path, args.dataset_name) + '/data.csv',
                                           args.num_clients)
    num_clients = args.num_clients

    # build model
    field_dims = dataset.field_dims
    numerical_num = dataset.numerical_num
    global_model = get_model(args.model_name, field_dims, numerical_num, args.task_num, args.expert_num, args.embed_dim)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    print_every = 5

    local_models = []
    for client_idx in range(args.num_clients):
        if args.separate_data:
            dataset_path = os.path.join(args.dataset_path, args.dataset_name) + f'/train_{client_idx}.csv'
            local_models.append(SeparatedDataLocalUpdate(args=args, data_path=dataset_path, field_dims=field_dims,
                                                         global_model=global_model, logger=logger))
        else:
            local_models.append(LocalUpdate(args=args, dataset=dataset, idxs=user_groups[client_idx],
                                            field_dims=field_dims, global_model=global_model, logger=logger))

    for epoch in tqdm(range(args.epoch)):
        local_weights, local_embedding_maps, local_update_times, local_train_nums, local_losses = [], [], [], [], []
        print(f'\n | Global Training Round : {epoch} |\n')

        global_model.train()
        m = max(int(args.frac * num_clients), 1)
        idxs_client = np.random.choice(list(range(num_clients)), m, replace=False)

        for idx in idxs_client:
            # TODO: in distributed computing, we need to pull personalized embedding for each client
            local_models[idx].fetch_embedding(global_model, False, lambda x: x > field_dims[0])  # pull embedding layer
            local_models[idx].fetch_weights(global_model)  # pull other layers
            w, loss = local_models[idx].update_weights()
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_update_times.append(copy.deepcopy(local_models[idx].update_times))
            local_train_nums.append(len(local_models[idx].dataset))
            local_embedding_maps.append(copy.deepcopy(local_models[idx].embedding_map))

        # average embedding weight except user_id embedding
        embedding_weight = average_embeddings(local_weights, local_embedding_maps, local_update_times, global_weights,
                                              args.embedding_name, lambda x: x > field_dims[0])

        # average global weights (do not consider the embedding layer)
        global_weights = average_weights(local_weights, local_train_nums, global_weights, args.embedding_name)

        # update global weights
        global_model.load_state_dict(global_weights)

        # update global embedding weight
        global_model.embedding.update_embedding(copy.deepcopy(embedding_weight))

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_eva, list_loss = [], []
        global_model.eval()
        for client_idx in range(num_clients):
            local_models[client_idx].fetch_embedding(global_model, False, lambda x: x > field_dims[0])
            local_models[client_idx].fetch_weights(global_model)
            eva, loss = local_models[client_idx].inference()
            list_eva.append(eva)
            list_loss.append(loss)
        train_evaluation = np.sum(list_eva, axis=0) / len(list_eva)

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            for i in range(args.task_num):
                print(f'Train {args.evaluation_name[i]} for task {i}: {train_evaluation[i]} \n')

    if args.separate_data:
        # Test inference after completion of training
        local_weights, local_embedding_maps, local_update_times = [], [], []
        for client_idx in range(args.num_clients):
            local_weights.append(local_models[client_idx].model.state_dict())
            local_embedding_maps.append(local_models[client_idx].embedding_map)
            local_update_times.append(copy.deepcopy(local_models[client_idx].update_times))

        # merge user_id embedding into global model
        embedding_weight = average_embeddings(local_weights, local_embedding_maps, local_update_times, global_weights,
                                              args.embedding_name, lambda x: x < field_dims[0])
        global_model.embedding.update_embedding(copy.deepcopy(embedding_weight))

        test_eva, test_loss = test_inference(args, global_model, dataset)
    else:
        # Test inference after completion of training
        local_weights, local_embedding_maps, local_update_times, test_idxs = [], [], [], []
        for user_id, idxs in user_groups.items():
            test_idxs.extend(local_models[user_id].idxs_test)
            local_weights.append(local_models[user_id].model.state_dict())
            local_embedding_maps.append(local_models[user_id].embedding_map)
            local_update_times.append(copy.deepcopy(local_models[user_id].update_times))

        # merge user_id embedding into global model
        embedding_weight = average_embeddings(local_weights, local_embedding_maps, local_update_times, global_weights,
                                              args.embedding_name, lambda x: x < field_dims[0])
        global_model.embedding.update_embedding(copy.deepcopy(embedding_weight))

        test_eva, test_loss = test_inference(args, global_model, DatasetSplit(dataset, test_idxs))

    print(f' \n Results after {args.epoch} global rounds of training:')
    print(f'Test Loss : {np.mean(np.array(test_loss))}')
    for i in range(args.task_num):
        print(f'|---- Test {args.evaluation_name[i]} for task {i}: {test_eva[i]}')

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
