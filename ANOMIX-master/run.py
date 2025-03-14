import torch
from model import Model
from utils import *
from sklearn.metrics import roc_auc_score, roc_curve
import random
import os
import dgl
import argparse
import time
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='ANOMIX')
parser.add_argument('--expid', type=int)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='ACM')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=100000) # cora:1000, citeseer:1110, pubmed:19718,
parser.add_argument('--subgraph_size', type=int, default=3)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=128)
parser.add_argument('--negsamp_ratio_patch', type=int, default=3)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.6, help='how much level info involves')
parser.add_argument('--beta', type=float, default=0.6, help='how much abnormality involves')
parser.add_argument('--ratio', type=int, default=10, help='anomaly ratio')
args = parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    print('Dataset: {}'.format(args.dataset), flush=True)
    print('alpha: {}'.format(args.alpha))
    print('beta: {}'.format(args.beta))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [i + 1 for i in range(args.runs)]

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    adj, features, label, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)
    nx_graph = nx.from_scipy_sparse_matrix(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    #nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    #labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    all_auc = []
    for run in range(args.runs):
        seed = seeds[run]
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context, args.readout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        for epoch in range(args.num_epoch):

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            anomaly_subgraphs = sampling_anomaly_subgraph(args.dataset)
            #anomaly_subgraphs = generate_anomaly_subgraph(nx_graph=nx_graph, label=label, ego_size=3, contam=0.1)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl_patch = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)
                lbl_context = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                # for normal
                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                # for abnormal
                a_ba = []
                a_bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, len(anomaly_subgraphs[0]))).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, len(anomaly_subgraphs[0]) + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in range(len(anomaly_subgraphs)):
                    cur_adj = adj[:, anomaly_subgraphs[i], :][:, :, anomaly_subgraphs[i]]
                    cur_feat = features[:, anomaly_subgraphs[i], :]
                    a_ba.append(cur_adj)
                    a_bf.append(cur_feat)

                ba_zero_rows = torch.zeros(cur_batch_size - len(a_ba), len(anomaly_subgraphs[0]), len(anomaly_subgraphs[0])).to(device)
                bf_zero_rows = torch.zeros(cur_batch_size - len(a_ba), len(anomaly_subgraphs[0]), features.size(2)).to(device)

                a_ba = torch.cat(a_ba)
                a_ba = torch.cat((a_ba, ba_zero_rows), dim=0)
                a_ba = torch.cat((a_ba, added_adj_zero_row), dim=1)
                a_ba = torch.cat((a_ba, added_adj_zero_col), dim=2)
                a_bf = torch.cat(a_bf)
                a_bf = torch.cat((a_bf, bf_zero_rows), dim=0)
                a_bf = torch.cat((a_bf[:, :-1, :], added_feat_zero_row, a_bf[:, -1:, :]), dim=1)

                logits_1, logits_2 = model(bf, ba)
                a_logits_1, a_logits_2 = model(a_bf, a_ba)

                # subgraph-level (normal)
                loss_all_1 = b_xent_context(logits_1, lbl_context)
                loss_1 = torch.mean(loss_all_1)

                # node-level (normal)
                loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                loss_2 = torch.mean(loss_all_2)

                # subgraph-level (abnormal)
                a_loss_all_1 = b_xent_context(a_logits_1, lbl_context)
                a_loss_1 = torch.mean(a_loss_all_1)

                # node-level (abnormal)
                a_loss_all_2 = b_xent_patch(a_logits_2, lbl_patch)
                a_loss_2 = torch.mean(a_loss_all_2)

                loss = (args.beta * (args.alpha * loss_1 + (1 - args.alpha) * loss_2) + (1 - args.beta) * (args.alpha * a_loss_1 + (1 - args.alpha) * a_loss_2)) / 2

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'checkpoints/exp_{}.pkl'.format(args.expid))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            #print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('checkpoints/exp_{}.pkl'.format(args.expid)))
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
        print('Testing AUC!', flush=True)

        for round in range(args.auc_test_rounds):
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            anomaly_subgraphs = sampling_anomaly_subgraph(args.dataset)
            #anomaly_subgraphs = generate_anomaly_subgraph(nx_graph=nx_graph, label=label, ego_size=3, contam=0.1)

            for batch_idx in range(batch_num):
                optimiser.zero_grad()
                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                # for normal test
                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                # for abnormal test
                a_ba = []
                a_bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, len(anomaly_subgraphs[0]))).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, len(anomaly_subgraphs[0]) + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in range(len(anomaly_subgraphs)):
                    cur_adj = adj[:, anomaly_subgraphs[i], :][:, :, anomaly_subgraphs[i]]
                    cur_feat = features[:, anomaly_subgraphs[i], :]
                    a_ba.append(cur_adj)
                    a_bf.append(cur_feat)

                ba_zero_rows = torch.zeros(cur_batch_size - len(a_ba), len(anomaly_subgraphs[0]), len(anomaly_subgraphs[0])).to(device)
                bf_zero_rows = torch.zeros(cur_batch_size - len(a_ba), len(anomaly_subgraphs[0]), features.size(2)).to(device)

                a_ba = torch.cat(a_ba)
                a_ba = torch.cat((a_ba, ba_zero_rows), dim=0)
                a_ba = torch.cat((a_ba, added_adj_zero_row), dim=1)
                a_ba = torch.cat((a_ba, added_adj_zero_col), dim=2)
                a_bf = torch.cat(a_bf)
                a_bf = torch.cat((a_bf, bf_zero_rows), dim=0)
                a_bf = torch.cat((a_bf[:, :-1, :], added_feat_zero_row, a_bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    test_logits_1, test_logits_2 = model(bf, ba)
                    a_test_logits_1, a_test_logits_2 = model(a_bf, a_ba)
                    test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))
                    a_test_logits_1 = torch.sigmoid(torch.squeeze(a_test_logits_1))
                    a_test_logits_2 = torch.sigmoid(torch.squeeze(a_test_logits_2))

                    test_logits_1 = (test_logits_1 + a_test_logits_1) / 2
                    test_logits_2 = (test_logits_2 + a_test_logits_2) / 2

                if args.alpha != 1.0 and args.alpha != 0.0:
                    if args.negsamp_ratio_context == 1 and args.negsamp_ratio_patch == 1:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                    ano_score = args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_2
                elif args.alpha == 1.0:
                    if args.negsamp_ratio_context == 1:
                        ano_score = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                elif args.alpha == 0.0:
                    if args.negsamp_ratio_patch == 1:
                        ano_score = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()

                multi_round_ano_score[round, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        auc = roc_auc_score(ano_label, ano_score_final)
        all_auc.append(auc)
        print('Testing AUC:{:.4f}'.format(auc), flush=True)

        #fpr, tpr, _ = roc_curve(ano_label, ano_score_final)
        #df_fpr = pd.DataFrame(fpr)
        #df_tpr = pd.DataFrame(tpr)
        #df_fpr.to_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/ROC/ACM/SGCL_fpr.csv', index=False, header=False)
        #df_tpr.to_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/ROC/ACM/SGCL_tpr.csv', index=False, header=False)

    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')
    print(f'{time.time() - start:.4f} sec')


