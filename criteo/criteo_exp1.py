
import numpy as np
from preprocess2 import get_criteo_data
import torch
from util2 import *
from mattack import pgd_attack
import argparse
from search_attack import grad_attack
from greedy_attack import greedy_attack

import torch.nn as nn


def main(args):
    X_train, X_test, y_train, y_test, num_names, cat_names, final_names = get_criteo_data(seed = args.seed)
    model = Net(X_train.shape[1]).cuda()
    model.load_state_dict(torch.load('./models/mlp1.pt'))

    ## doing adversarial attack
    if args.svd == 'load':
        s = np.load('./svd_logs/s_criteo.npy')
        Vh = np.load('./svd_logs/Vh_criteo.npy')
    elif args.svd == 'mixed':
        c, s, Vh = svd_mixed(X_train, len(num_names), dataset= 'criteo')
    else:
        raise ValueError

    ## doing adversarial attack
    neg_ind = np.where(y_test == 1)[0]
    batch_size = 1
    test_set_size = 50
    m = args.m_weight

    ## do mixed pgd attack
    configs2 = {'s':s, 'Vh': Vh, 'num_steps': 30, 'eps1': args.eps1, 'eps2': args.eps2, 'k_num': len(num_names)}

    def get_stat(sample, dat, target, m):
        criterion = nn.BCELoss()
        s_inv = np.diag(1 / s)
        M = torch.tensor((Vh.T).dot(s_inv), dtype=torch.float).cuda()
        M_dist = torch.norm((torch.matmul(sample, M) - torch.matmul(dat, M)))

        output = model(sample)
        loss = criterion(torch.sigmoid(output).flatten(), target)
        overall = loss - m * M_dist
        return loss.item(), M_dist.item(), overall.item()

    all_loss_mattack = []
    all_dist_mattack = []
    all_loss_search = []
    all_dist_search = []
    all_loss_greedy = []
    all_dist_greedy = []
    t = 0
    for i in range((int(len(neg_ind) / batch_size) + 1)):
            data = torch.tensor(X_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
            target = torch.tensor(y_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
            output = model(data)
            pred = (output >= 0.0)

            if pred.item() == True:
                #################################################################################
                ## greedy attack    ## only record valid attacks
                sample_x = greedy_attack(model, data, target, cat_names, final_names, m_weight = m, **configs2)
                loss1, m_dist1, _ = get_stat(sample_x, data, target, m)
                all_loss_greedy.append(loss1)
                all_dist_greedy.append(m_dist1)

                #################################################################################
                e_list = [0.1, 0.2, 0.4, 0.6, 0.8]
                w_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.5]
                all_stat = []
                loss, m_dist, overall = get_stat(data, data, target, m) ## if not sucessful, not attack
                all_stat.append([loss, m_dist, overall])
                for ee in e_list:
                    for cw in w_list:
                        configs1 = {'s': s, 'Vh': Vh, 'num_steps': 50, 'eps1': args.eps1, 'eps2s': ee, 'ce_weight': cw}
                        sample_x = pgd_attack(model, data, target, num_names, cat_names, final_names, m_weight = m, **configs1)
                        for j in range(sample_x.shape[0]):
                            sample = sample_x[j:j + 1]
                            l1_dist, cat_perb = check_attack(sample, data, num_names)
                            if cat_perb <= args.eps2:
                                loss, m_dist, overall = get_stat(sample, data, target, m)
                                all_stat.append([loss, m_dist, overall])
                all_stat = np.array(all_stat)
                best_idx = np.argmax(all_stat[:,2])
                loss_best, m_dist_best, _ = all_stat[best_idx]
                all_loss_mattack.append(loss_best)
                all_dist_mattack.append(m_dist_best)

                #################################################################################
                ## search attack    ## only record valid attacks
                sample_x = grad_attack(model, data, target, cat_names, final_names, m_weight = m, **configs2)
                loss, m_dist, _ = get_stat(sample_x, data, target, m)
                all_loss_search.append(loss)
                all_dist_search.append(m_dist)
                print('Aeverage of ' + str(t) + ' samples ', loss1, m_dist1, loss, m_dist, loss_best, m_dist_best, flush = True)
                t = t + 1
                if t > test_set_size:
                    break

    avg_loss_search = np.mean(np.array(all_loss_search))
    avg_dist_search = np.mean(np.array(all_dist_search))
    avg_loss_mattack = np.mean(np.array(all_loss_mattack))
    avg_dist_mattack = np.mean(np.array(all_dist_mattack))
    avg_loss_greedy = np.mean(np.array(all_loss_greedy))
    avg_dist_greedy = np.mean(np.array(all_dist_greedy))

    print('..............................................')
    print('final results')
    print(avg_loss_greedy, avg_dist_greedy, avg_loss_search, avg_dist_search, avg_loss_mattack, avg_dist_mattack)
    record = np.array([avg_loss_greedy, avg_dist_greedy, avg_loss_search, avg_dist_search, avg_loss_mattack, avg_dist_mattack])

    with open('record_' +str(args.eps1) + '_' +str(args.eps2) + '.txt', 'a') as f:
        f.write(np.array2string(record, precision=5, separator=',') + '\n')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--svd', type= str, help = 'whether load existing svd', default= 'load')
    argparser.add_argument('--eps1', type=float, help='bound of numerical features', default= 0.5)
    argparser.add_argument('--eps2', type=int, help='bound of categorical features', default= 3)
    argparser.add_argument('--m-weight', type=float, help= 'M-Distance Penalty', default= 2.0)
    args = argparser.parse_args()
    print('Criteo Display Advertisement')
    print(args)
    main(args)
