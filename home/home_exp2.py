import numpy as np
import pandas as pd
import os
import argparse
from home_preprocess import *
import torch
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mattack import pgd_attack
from search_attack import grad_attack
from greedy_attack import greedy_attack
import time

def main(args):

    ## get train/test data, X is N by 62
    X_train, X_test, y_train, y_test, num_names, cat_names, final_names = get_home_data(seed = args.seed)

    ## train model
    model = Net2(X_train.shape[1]).cuda()
    model.load_state_dict(torch.load('./models/mlp1.pt'))

    if args.svd == 'load':
        s = np.load('./svd_logs/s_home.npy')
        Vh = np.load('./svd_logs/Vh_home.npy')
    elif args.svd == 'mixed':
        c, s, Vh = svd_mixed(X_train, len(num_names), dataset= 'home')
    else:
        raise ValueError


    ## doing adversarial attack
    neg_ind = np.where(y_test == 1)[0]
    batch_size = 1
    test_set_size = 200

    t = 0
    sr1 = 0
    sr2 = 0
    sr3 = 0
    time1 = 0
    time2 = 0
    time3 = 0

    ## do mixed pgd attack
    configs2 = {'s':s, 'Vh': Vh, 'num_steps': 30, 'eps1': args.eps1, 'eps2': args.eps2, 'k_num': len(num_names), 'm_weight': args.m_weight}

    for i in range((int(len(neg_ind) / batch_size) + 1)):

        data = torch.tensor(X_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
        target = torch.tensor(y_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
        output = model(data)
        pred = (output > 0.0)

        if pred.item() == True:     ## only attack the positive samples

            ###################################################### M-Attack
            e_list = [0.1, 0.2, 0.4, 0.6, 0.8]
            w_list = [0.2, 0.5, 0.8, 1.5, 3.0, 5.0, 10.0]
            start_time1 = time.time()
            total = 0
            for ee in e_list:
                for cw in w_list:
                    configs1 = {'s': s, 'Vh': Vh, 'num_steps': 30, 'eps1': args.eps1, 'eps2s': ee, 'ce_weight': cw}
                    sample_x = pgd_attack(model, data, target, num_names, cat_names, final_names, **configs1)
                    for j in range(sample_x.shape[0]):
                        sample = sample_x[j:j + 1]
                        l1_dist, cat_perb = check_attack(sample, data, num_names)
                        if ((cat_perb <= args.eps2) & (model(sample).item() < - 0.2)):
                            sr1 = sr1 + 1
                            total = total + 1
                            break
                    if total > 0:
                        break
                if total > 0:
                    break
            endtime1 = time.time()
            time1 = time1 + (endtime1 - start_time1)

            ###################################################### Search-Attack
            start_time2 = time.time()
            sample_x = grad_attack(model, data, target, cat_names, final_names, **configs2)
            output = model(sample_x)
            endtime2 = time.time()
            time2 = time2 + (endtime2 - start_time2)
            if (output < - 0.2): sr2 = sr2 + 1

            ###################################################### Search-Attack
            start_time3 = time.time()
            sample_x = greedy_attack(model, data, target, cat_names, final_names, **configs2)
            output = model(sample_x)
            endtime3 = time.time()
            time3 = time3 + (endtime3 - start_time3)
            if (output < - 0.2): sr3 = sr3 + 1

            ####################################################### report performance
            t = t + 1
            if t % 5 == 0:
                print('successful rate', np.round(sr1 / t, 3), np.round(sr2 / t, 3), np.round(sr3 / t, 3), flush=True)
                print('time', np.round(time1 / t, 3), np.round(time2 / t, 3), np.round(time3 / t, 3), flush=True)
                print('.................')
            if t > test_set_size:
                break


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--svd', type= str, help = 'whether load existing svd', default= 'load')
    argparser.add_argument('--eps1', type=float, help='bound of numerical features', default=0.5)
    argparser.add_argument('--eps2', type=int, help='bound of categorical features', default= 3)
    argparser.add_argument('--m-weight', type=float, help= 'M-Distance Penalty', default= 2.0)
    args = argparser.parse_args()
    print('Home Credit')
    print(args)
    main(args)
