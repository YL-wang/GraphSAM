import argparse
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from dataset_graph import construct_dataset, mol_collate_func
from transformer_graph import make_model
from utils import ScheduledOptim, get_options, get_loss, cal_loss, evaluate, scaffold_split, build_lr_scheduler
from collections import defaultdict
from scheduler import NoamLR
from sam import SAM, GraphSAM, LookSAM
import warnings
warnings.filterwarnings("ignore")
import copy
model_initial = 0
model_final = 0

def model_train(model, train_dataset, valid_dataset, model_params, train_params, dataset_name, fold):
    # build data loader
    # stores the initial point in parameter space
    model_initial = copy.deepcopy(model)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # build loss function
    criterion = get_loss(train_params['loss_function'])


    # build optimizer

    if args.sam == 1:
        print('SAM')
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.init_lr,  weight_decay=args.weight_decay)
        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

    elif args.sam == 2:
        print('GraphSam')
        base_optimizer = torch.optim.Adam
        optimizer = GraphSAM(model.parameters(), base_optimizer, arg=args, lr=args.init_lr,
                                            weight_decay=args.weight_decay)
        # optimizer = ScheduledOptim(GraphSAM(model.parameters(), base_optimizer, arg=args, lr=args.init_lr,
        #                                             weight_decay=args.weight_decay),
        #                            train_params['warmup_factor'], model_params['d_model'],
        #                            train_params['total_warmup_steps'])
        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)
    else:
        print('Adam')
        optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                                   train_params['warmup_factor'], model_params['d_model'],
                                   train_params['total_warmup_steps'])

    # optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
    #                            train_params['warmup_factor'], model_params['d_model'],
    #                            train_params['total_warmup_steps'])

    best_valid_metric = float('inf') if train_params['task'] == 'regression' else float('-inf')
    best_epoch = -1
    best_valid_result, best_valid_bedding = None, None

    gh_list = []
    gs_list = []
    for epoch in range(train_params['total_epochs']):
        # train
        train_loss = list()
        model.train()
        i=0
        if args.sam == 1:
            for batch in tqdm(train_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
                y_true = y_true.to(train_params['device'])  # (batch, task_numbers)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                # (batch, task_numbers)

                ##################
                def closure():
                    y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
                    loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                                    train_params['mean'], train_params['std'], train_params['device'])
                    loss.backward(loss.clone().detach())
                    return loss
                ##################
                y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
                loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                                train_params['mean'], train_params['std'], train_params['device'])

                loss.backward()
                # for name, parms in model.named_parameters():
                #
                #     if name == 'encoder.layers.1.self_attn.linears.0.weight':
                #         gh = parms.grad[0].cpu().numpy()
                #         gh_list.append(gh)

                optimizer.step(i, epoch, closure)
                # for name, parms in model.named_parameters():
                #
                #     if name == 'encoder.layers.1.self_attn.linears.0.weight':
                #         gs = parms.grad[0].cpu().numpy()
                #         gs_list.append(gs)

                optimizer.zero_grad()

                if isinstance(scheduler, NoamLR):
                    scheduler.step()
                train_loss.append(loss.detach().item())
                i = i + 1
        elif args.sam == 2:
            for batch in tqdm(train_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
                y_true = y_true.to(train_params['device'])  # (batch, task_numbers)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                # (batch, task_numbers)
                ##################
                def closure():
                    y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
                    loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                                    train_params['mean'], train_params['std'], train_params['device'])
                    loss.backward(loss.clone().detach())
                    return loss
                ##################
                if i == 0:
                    y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
                    loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                                    train_params['mean'], train_params['std'], train_params['device'])
                    optimizer.step(i, epoch, closure, loss)
                else:
                    optimizer.step(i, epoch, closure)

                # for name, parms in model.named_parameters():
                #
                #     if name == 'encoder.layers.1.self_attn.linears.0.weight':
                #         gs = parms.grad[0].cpu().numpy()
                #         gs_list.append(gs)

                loss = optimizer.get_loss()
                optimizer.zero_grad()

                if isinstance(scheduler, NoamLR):
                    scheduler.step()
                train_loss.append(loss.detach().item())
                i = i + 1
        else:
            for batch in tqdm(train_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
                y_true = y_true.to(train_params['device'])  # (batch, task_numbers)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                # (batch, task_numbers)

                y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
                loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                                train_params['mean'], train_params['std'], train_params['device'])
                optimizer.zero_grad()
                loss.backward()

                # for name, parms in model.named_parameters():
                    # print(name)
                    # import ipdb
                    # ipdb.set_trace()
                    #
                    #     # if name=='mol_atom_from_atom_ffn.1.weight':
                    #     #     g=parms.grad[0].cpu()
                    #     #     g1 = g-g1
                    #     #     g_norm = torch.norm(g1)
                    #     #
                    #     #     with open('/root/grover-main/g3.txt', 'a', encoding='utf-8') as f:
                    #     #        f.write("%.4f"%g_norm)
                    #     #        f.write(',')
                    #
                    # if name == 'encoder.layers.1.self_attn.linears.0.weight':
                    #     gh = parms.grad[0].cpu().numpy()
                    #     gh_list.append(gh)


                optimizer.step_and_update_lr()
                train_loss.append(loss.detach().item())

        # valid
        model.eval()
        with torch.no_grad():
            valid_true, valid_pred, valid_smile, valid_embedding = list(), list(), list(), list()
            for batch in tqdm(valid_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                # (batch, task_numbers)
                y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features)

                y_true = y_true.numpy()  # (batch, task_numbers)
                y_pred = y_pred.detach().cpu().numpy()  # (batch, task_numbers)
                y_embedding = y_embedding.detach().cpu().numpy()

                valid_true.append(y_true)
                valid_pred.append(y_pred)
                valid_smile.append(smile_list)
                valid_embedding.append(y_embedding)

            valid_true, valid_pred = np.concatenate(valid_true, axis=0), np.concatenate(valid_pred, axis=0)
            valid_smile, valid_embedding = np.concatenate(valid_smile, axis=0), np.concatenate(valid_embedding, axis=0)

        valid_result = evaluate(valid_true, valid_pred, valid_smile,
                                requirement=['sample', train_params['loss_function'], train_params['metric']],
                                data_mean=train_params['mean'], data_std=train_params['std'], data_task=train_params['task'])

        # save and print message in graph regression
        if train_params['task'] == 'regression':
            if valid_result[train_params['metric']] < best_valid_metric:
                best_valid_metric = valid_result[train_params['metric']]
                best_epoch = epoch + 1
                best_valid_result = valid_result
                best_valid_bedding = valid_embedding
                torch.save({'state_dict': model.state_dict(),
                            'best_epoch': best_epoch,
                            f'best_valid_{train_params["metric"]}': best_valid_metric},
                           f'./Model/{dataset_name}/best_model_{dataset_name}_fold_{fold}.pt')

            # print("Epoch {}, learning rate {:.6f}, "
            #       "train {}: {:.4f}, "
            #       "valid {}: {:.4f}, "
            #       "best epoch {}, best valid {}: {:.4f}"
            #       .format(epoch + 1, scheduler.get_lr()[-1],
            #               train_params['loss_function'], np.mean(train_loss),
            #               train_params['loss_function'], valid_result[train_params['loss_function']],
            #               best_epoch, train_params['metric'], best_valid_metric
            #               ))

            print("Epoch {}, "
                  "train {}: {:.4f}, "
                  "valid {}: {:.4f}, "
                  "best epoch {}, best valid {}: {:.4f}"
                  .format(epoch + 1,
                          train_params['loss_function'], np.mean(train_loss),
                          train_params['loss_function'], valid_result[train_params['loss_function']],
                          best_epoch, train_params['metric'], best_valid_metric
                          ))

        # save and print message in graph classification
        else:
            if valid_result[train_params['metric']] > best_valid_metric:
                best_valid_metric = valid_result[train_params['metric']]
                best_epoch = epoch + 1
                best_valid_result = valid_result
                best_valid_bedding = valid_embedding
                torch.save({'state_dict': model.state_dict(),
                            'best_epoch': best_epoch,
                            f'best_valid_{train_params["metric"]}': best_valid_metric},
                           f'./Model/{dataset_name}/best_model_{dataset_name}_fold_{fold}.pt')

            # print("Epoch {}, learning rate {:.6f}, "
            #       "train {}: {:.4f}, "
            #       "valid {}: {:.4f}, "
            #       "valid {}: {:.4f}, "
            #       "best epoch {}, best valid {}: {:.4f}"
            #       .format(epoch + 1, scheduler.get_lr()[-1],
            #               train_params['loss_function'], np.mean(train_loss),
            #               train_params['loss_function'], valid_result[train_params['loss_function']],
            #               train_params['metric'], valid_result[train_params['metric']],
            #               best_epoch, train_params['metric'], best_valid_metric
            #               ))

            print("Epoch {}, "
                  "train {}: {:.4f}, "
                  "valid {}: {:.4f}, "
                  "valid {}: {:.4f}, "
                  "best epoch {}, best valid {}: {:.4f}"
                  .format(epoch + 1,
                          train_params['loss_function'], np.mean(train_loss),
                          train_params['loss_function'], valid_result[train_params['loss_function']],
                          train_params['metric'], valid_result[train_params['metric']],
                          best_epoch, train_params['metric'], best_valid_metric
                          ))

        # early stop
        if abs(best_epoch - epoch) >= 20:
            print("=" * 20 + ' early stop ' + "=" * 20)
            break
        # import ipdb
        # ipdb.set_trace()
        # def Save_list(list1, filename):
        #     file2 = open(filename , 'w')
        #     for i in range(len(list1)):
        #         for j in range(len(list1[i])):
        #             file2.write(str(list1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
        #             file2.write('	')  # 相当于Tab一下，换一个单元格
        #         file2.write('\n')                                   # 写完一行立马换行
        #     file2.close()
        # if epoch == 19:
        #     if args.sam == 1:
        #         file1 = '/home/wangyili/G2G/gs_sam.txt'
        #         Save_list(gs_list, file1)
        #
        #         file2 = '/home/wangyili/G2G/gh_sam.txt'
        #         Save_list(gh_list, file2)
        #     elif args.sam ==2:
        #         # file1 = '/home/wangyili/G2G/gh_graphsam.txt'
        #         # Save_list(gh_list, file1)
        #         file2 = '/home/wangyili/G2G/gs_graphsam.txt'
        #         Save_list(gs_list, file2)
        #     else:
        #         file = '/home/wangyili/G2G/gh_adam.txt'
        #         Save_list(gh_list, file)

        with open('/root/grover-main/train_loss.txt', 'a', encoding='utf-8') as f:
           f.write("%.4f"%np.mean(train_loss))
           f.write(',')



    return best_valid_result, best_valid_bedding


def model_test(checkpoint, test_dataset, model_params, train_params):
    # build loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                             shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    # build model
    model = make_model(**model_params)
    model.to(train_params['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model_final = copy.deepcopy(model)
    # test
    model.eval()
    with torch.no_grad():
        test_true, test_pred, test_smile, test_embedding = list(), list(), list(), list()
        for batch in tqdm(test_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            # (batch, task_numbers)
            y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features)

            y_true = y_true.numpy()  # (batch, task_numbers)
            y_pred = y_pred.detach().cpu().numpy()  # (batch, task_numbers)
            y_embedding = y_embedding.detach().cpu().numpy()

            test_true.append(y_true)
            test_pred.append(y_pred)
            test_smile.append(smile_list)
            test_embedding.append(y_embedding)
        test_true, test_pred = np.concatenate(test_true, axis=0), np.concatenate(test_pred, axis=0)
        test_smile, test_embedding = np.concatenate(test_smile, axis=0), np.concatenate(test_embedding, axis=0)

    test_result = evaluate(test_true, test_pred, test_smile,
                           requirement=['sample', train_params['loss_function'], train_params['metric']],
                           data_mean=train_params['mean'], data_std=train_params['std'], data_task=train_params['task'])

    print("test {}: {:.4f}".format(train_params['metric'], test_result[train_params['metric']]))

    return test_result, test_embedding


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu', default=-1)
    parser.add_argument("--fold", type=int, help='the number of k-fold', default=5)
    parser.add_argument("--dataset", type=str, help='choose a dataset', default='esol')
    parser.add_argument("--split", type=str, help="choose the split type", default='random',
                        choices=['random', 'scaffold', 'cv'])

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to task')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')


    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')


    parser.add_argument('--sam', type=int, default=0,
                        help='Use sam 0 no use')
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--radius', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--epoch_steps', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.5)

    args = parser.parse_args()

    # load options
    model_params, train_params = get_options(args.dataset)

    # init device and seed
    print(f"Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    # load data
    if train_params['task'] == 'regression':
        with open(f'./Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label, data_mean, data_std] = pkl.load(f)
    else:
        with open(f'./Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label] = pkl.load(f)

    # calculate the padding
    model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])
    print(f"Max padding length is: {model_params['max_length']}")

    # construct dataset
    print('=' * 20 + ' construct dataset ' + '=' * 20)
    dataset = construct_dataset(data_mol, data_label, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])
    total_metrics = defaultdict(list)

    # split dataset
    if args.split == 'scaffold':
        # we run the scaffold split 5 times for different random seed, which means different train/valid/test
        for idx in range(args.fold):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # get dataset
            train_index, valid_index, test_index = scaffold_split(data_mol, frac=[0.8, 0.1, 0.1], balanced=True,
                                                                  include_chirality=False, ramdom_state=args.seed + idx)
            train_dataset, valid_dataset, test_dataset = dataset[train_index], dataset[valid_index], dataset[test_index]

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            args.steps = int(len(train_dataset) / train_params['batch_size'])
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
            best_valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, idx + 1)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])

            # test
            print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.dataset}_fold_{idx + 1}.pt', map_location=train_params['device'])
            test_result, test_embedding = model_test(checkpoint, test_dataset, model_params, train_params)
            test_csv = pd.DataFrame.from_dict({'smile': test_result['smile'], 'actual': test_result['label'], 'predict': test_result['prediction']})
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(test_result['smile'], test_embedding):
                total_embedding[smile] = embedding

            with open(f'./Result/{args.dataset}/total_test_embedding_fold_{idx + 1}.pickle', 'wb') as fw:
                pkl.dump(total_embedding, fw)

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for idx in range(args.fold):
            print('fold {}, valid {} = {:.4f}, test {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['valid'][idx],
                          train_params['metric'], total_metrics['test'][idx]))

        print('{} folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

    elif args.split == 'random':
        # we run the random split 5 times for different random seed, which means different train/valid/test
        for idx in range(args.fold):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # get dataset
            # train_valid_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=args.seed)
            # train_dataset, valid_dataset = train_test_split(train_valid_dataset, test_size=len(test_dataset),
            #                                                 random_state=args.seed)
            index=[]
            for i in range(len(dataset)):
                index.append(i)
            train_index, test_index = train_test_split(index, test_size=0.1, random_state=args.seed)
            train_index, valid_index = train_test_split(train_index, test_size=len(test_index),
                                                            random_state=args.seed)

            train_dataset, valid_dataset, test_dataset = dataset[train_index], dataset[valid_index], dataset[test_index]

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            args.steps = int(len(train_dataset) / train_params['batch_size'])
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':

                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")


            best_valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, idx + 1)
            best_valid_csv = pd.DataFrame.from_dict({'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])

            # test
            print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.dataset}_fold_{idx + 1}.pt')
            test_result, _ = model_test(checkpoint, test_dataset, model_params, train_params)
            test_csv = pd.DataFrame.from_dict({'actual': test_result['label'], 'predict': test_result['prediction']})
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for idx in range(args.fold):
            print('fold {}, valid {} = {:.4f}, test {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['valid'][idx],
                          train_params['metric'], total_metrics['test'][idx]))

        print('{} folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

    elif args.split == 'cv':
        # k-fold
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(X=dataset)):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # get dataset
            train_dataset, valid_dataset = dataset[train_index], dataset[valid_index]

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}")
            best_valid_result, best_valid_embedding = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, idx + 1)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['cv'].append(best_valid_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(best_valid_result['smile'], best_valid_embedding):
                total_embedding[smile] = embedding

            with open(f'./Result/{args.dataset}/total_valid_embedding_fold_{idx + 1}.pickle', 'wb') as fw:
                pkl.dump(total_embedding, fw)

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for idx in range(args.fold):
            print('fold {} {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['cv'][idx]))
        print('{} folds {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['cv']), np.nanstd(total_metrics['cv'])))
        print('=' * 20 + " finished! " + '=' * 20)
    else:
        raise Exception('We only support random split, scaffold split and cv!')
