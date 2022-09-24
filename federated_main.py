import copy
import os
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from util.options import args_parser
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
    dataset, user_groups = get_dataset(args.dataset_name,
                                       os.path.join(args.dataset_path, args.dataset_name) + '/data.csv',
                                       args.num_users)
    user_num = len(user_groups.keys())

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

    for epoch in tqdm(range(args.epoch)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * user_num), 1)
        idxs_users = np.random.choice(list(user_groups.keys()), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model).to(device), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # average embedding weight
        embedding_weight = average_embeddings(local_weights, global_weights, args.embedding_name)

        # average global weights
        global_weights = average_weights(local_weights, args.embedding_name)

        # update global embedding weight
        global_model.state_dict()[args.embedding_name] = copy.deepcopy(embedding_weight)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_eva, list_loss = [], []
        global_model.eval()
        for c in range(user_num):
            local_model = LocalUpdate(args=args, dataset=dataset,
                                      idxs=user_groups[c], logger=logger)
            eva, loss = local_model.inference(model=global_model)
            list_eva.append(eva)
            list_loss.append(loss)
        train_evaluation = np.sum(list_eva, axis=0) / len(list_eva)

        # print global training loss after every 'i' rounds
        if epoch % print_every == 0:
            print(f' \nAvg Training Stats after {epoch} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            for i in range(args.task_num):
                print(f'Train {args.evaluation_name[i]} for task {i}: {train_evaluation[i]} \n')

    # Test inference after completion of training
    test_idxs = []
    for idxs in user_groups.values():
        test_idxs.extend(idxs[int(0.8 * len(idxs)):])
    test_eva, test_loss = test_inference(args, global_model, DatasetSplit(dataset, test_idxs))

    print(f' \n Results after {args.epoch} global rounds of training:')
    for i in range(args.task_num):
        print(f'|---- Avg Train {args.evaluation_name[i]} for task {i}: {train_evaluation[i]}')
        print(f'|---- Test {args.evaluation_name[i]} for task {i}: {test_eva[i]}')

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

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
