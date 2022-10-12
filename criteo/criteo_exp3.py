
import numpy as np
from preprocess2 import get_criteo_data
import torch
from util2 import *
from mattack import pgd_attack
import argparse
from search_attack import grad_attack
from greedy_attack import greedy_attack

import torch.nn as nn
import time
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def main(args):
    X_train, X_test, y_train, y_test, num_names, cat_names, final_names = get_criteo_data(seed = args.seed)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_train[0:30000])
    aa = (kde.score_samples(X_test[0:300]))
    bins = np.arange(np.min(aa) + 65, np.max(aa), 5)
    print(np.min(aa), np.max(aa))
    plt.xticks(bins)

    model = Net(X_train.shape[1]).cuda()
    model.load_state_dict(torch.load('./models/mlp1.pt'))

    ## doing adversarial attack
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

    all_aa_c = []
    all_aa_0 = []
    all_aa_5 = []

    t = 0

    for i in range((int(len(neg_ind) / batch_size) + 1)):

        data = torch.tensor(X_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
        target = torch.tensor(y_test[neg_ind[i * batch_size: (i + 1) * batch_size]], dtype=torch.float).cuda()
        output = model(data)
        pred = (output > 0.0)

        if pred.item() == True:  ## only attack the positive samples
            aa = kde.score_samples(data.detach().cpu().numpy())
            all_aa_c.append(aa)

            configs1 = {'s': s, 'Vh': Vh, 'num_steps': 20, 'eps1': args.eps1, 'eps2s': 0.6, 'ce_weight': 0.3, 'm_num': 1}
            sample_x = pgd_attack(model, data, target, num_names, cat_names, final_names, m_weight= 0, **configs1)
            aa = kde.score_samples(sample_x.detach().cpu().numpy())
            all_aa_0.append(aa)

            configs1 = {'s': s, 'Vh': Vh, 'num_steps': 20, 'eps1': args.eps1, 'eps2s': 0.6, 'ce_weight': 0.3, 'm_num': 1}
            sample_x = pgd_attack(model, data, target, num_names, cat_names, final_names, m_weight= 6, **configs1)
            aa = kde.score_samples(sample_x.detach().cpu().numpy())
            all_aa_5.append(aa)

            t = t + 1
            if t % 10 == 0:
                print(t)
            if t > 100:
                break

    all_aa_c = np.array(all_aa_c).flatten()
    all_aa_0 = np.array(all_aa_0).flatten()
    all_aa_5 = np.array(all_aa_5).flatten()
    all_aa = np.array([all_aa_0, all_aa_5, all_aa_c]).T
    colors = ['red', 'blue', 'green']
    labels = ['M-Dist. Reg. = 0', 'M-Dist. Reg. = 6' , 'Clean Sample']

    plt.hist(all_aa, bins=bins, color = colors, density = True, label = labels)
    plt.legend(fontsize = 12, loc = 2)
    plt.xlabel('Log-Likelihood', fontsize = 12)
    plt.savefig('123.png')


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
