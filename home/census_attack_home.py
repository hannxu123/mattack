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
from linear_programming33 import *
from mixed_attack2 import pgd_attack
import time

def main(args):

    ## get train/test data, X is N by 62
    X_train, X_test, y_train, y_test, num_names, cat_names, final_names = get_home_data(seed = args.seed)

    ## get location of categorical features
    all_cat_i = []
    all_cat_j = []
    for j in range(len(cat_names)):
        for i in range(len(final_names)):
            if cat_names[j] in final_names[i]:
                if j not in all_cat_j:
                    all_cat_j.append(j)
                    all_cat_i.append(i)
    all_cat_i.append(X_train.shape[1])

    ## train model
    model = Net2(X_train.shape[1]).cuda()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 1e-3)
    #optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)
    #train(model, X_train, y_train,  X_test, y_test, optimizer, optimizer_scheduler)
    #torch.save(model.state_dict(), './checkpoints/home_mlp2.pt')
    model.load_state_dict(torch.load('./checkpoints/home_mlp2.pt'))

    ## doing adversarial attack
    if args.svd == 'load':
        X2 = X_train[:, len(num_names):]
        c = np.expand_dims(np.mean(X2, axis=0), 0)
        s = np.load('svd_logs/s_home.npy')
        Vh = np.load('svd_logs/Vh_home.npy')
    elif args.svd == 'mixed':
        c, s, Vh = svd_mixed(X_train, len(num_names))
    else:
        raise ValueError

    ## doing adversarial attack
    neg_ind = np.where(y_test == 1)[0]
    batch_size = 1
    test_set_size = 200

    t = 0
    sr1 = 0
    sr2 = 0
    time1 = 0
    time2 = 0

    ## do mixed pgd attack
    configs = {'c': c,
               's': s,
               'Vh': Vh,
               'num_steps': 30,
               'Lambda': 0.0,
               'eps1': args.eps1,
               }

    for i in range((int(len(neg_ind) / batch_size) + 1)):

        data = torch.tensor(X_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
        target = torch.tensor(y_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
        output = model(data)
        pred = (output > 0.0)

        alpha_list = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 8.0]

        if pred.item() == True:
            t_total = 0
            start_time = time.time()
            for aa in alpha_list:
                sample_x = pgd_attack(model, data, target, num_names, cat_names, final_names, lam_ce=aa, **configs)
                for i in range(sample_x.shape[0]):
                    sample = sample_x[i:i + 1]
                    l1_dist, cat_perb = check_attack(sample, data, num_names, final_names, verbose=False)
                    if ((cat_perb <= args.eps2) & (model(sample).item() < 0.0)):
                        sr1 = sr1 + 1
                        t_total = t_total + 1
                        break

                if t_total > 0:
                    endtime = time.time()
                    break
                else:
                    endtime = time.time()

            time1 = time1 + (endtime - start_time)

            ## do gradient guided attack
            starttime = time.time()
            sample_x = grad_attack(model, data, target, 5,
                                   cat_names, final_names, num_steps=20, eps1=args.eps1, eps2=args.eps2)
            output = model(sample_x)
            endtime = time.time()
            time2 = time2 + (endtime - start_time)

            if (output < 0.0):
                sr2 = sr2 + 1

            t = t + 1
            if t % 10 == 0:
                print('successful rate', np.round(sr1 / t, 3), np.round(sr2 / t, 3), flush=True)
                print('time', np.round(time1 / t, 3), np.round(time2 / t, 3), flush=True)
                print('.................')
            if t > test_set_size:
                break
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--svd', type= str, help = 'whether load existing svd', default= 'load')
    argparser.add_argument('--eps1', type=float, help='bound of numerical features', default=0.5)
    argparser.add_argument('--eps2', type=int, help='bound of categorical features', default= 2)
    argparser.add_argument('--Lambda', type=float, help= 'M-Distance Penalty', default= 0.0)
    args = argparser.parse_args()
    print('Home Credet')
    print(args)
    main(args)