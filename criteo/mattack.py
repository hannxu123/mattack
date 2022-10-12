import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util2 import get_gumbel_output2, get_gumbel_output3, get_ce_loss
import cvxpy as cp


def linear_proj_num(z_p, x, eps1 = 1.0):

    k = x.shape[1]

    # Problem data.
    x1 = cp.Variable((1, k))

    # objective
    objective = cp.Minimize(cp.sum_squares(x1 - z_p))   ## z_p is the current attack, not not eligible

    # define constraint
    all_constraint = [cp.sum(cp.abs(x1 - x)) <= eps1]
    prob = cp.Problem(objective, all_constraint)

    try:
        prob.solve()
        return x1.value
    except:
        return z_p

def pgd_attack(model, x, y,
               num_features,
               cat_features,
               all_features,
               s=None, Vh=None,
               num_steps = 20,
               eps2s = 0.01, ce_weight = 0.01,
               eps1 = 0.5, m_weight = 2, m_num = 1000):

    ## get the locations of encoded representation
    all_cat_i = []
    all_cat_j = []

    for j in range(len(cat_features)):
        for i in range(len(all_features)):
            if (cat_features[j] + '_') in all_features[i]:
                if j not in all_cat_j:
                    all_cat_j.append(j)
                    all_cat_i.append(i)
    all_cat_i.append(x.shape[1])
    cat_list = all_cat_i

    ## attack setups
    criterion = nn.BCELoss(reduction = 'none')
    k_num = len(num_features)

    ## initalize the z point based on x
    z = torch.tensor(x, dtype=torch.float).cuda()
    z[:,k_num:] = 15 * z[:,k_num:]

    z.requires_grad = True

    ## prepare for Mahalanobis Distance
    s_inv = np.diag(1 / s)
    M = torch.tensor((Vh.T).dot(s_inv), dtype = torch.float).cuda()

    for j in range(num_steps):

        sample_x = get_gumbel_output2(z, number= 500, tau = 0.1, cat_list= cat_list)
        output = model(sample_x)

        ## Loss + M-distance
        aa = torch.matmul(sample_x, M)
        bb = torch.matmul(x, M).repeat(500, 1)
        M_dist = torch.norm((aa - bb), dim= 1)  # averaged M distance
        loss = torch.mean(criterion(torch.sigmoid(output), y.repeat(500, 1))) - m_weight * torch.mean(M_dist)

        ## categorical constraint
        ce_loss = F.relu(get_ce_loss(z, x, cat_list) - eps2s)
        obj_loss = loss - ce_loss * ce_weight
        grad = torch.autograd.grad(obj_loss, z)[0]

        ## update and project numerical feature
        z_num = z[:, 0:k_num] + 0.05 * grad[:, 0:k_num]
        z_p_num = linear_proj_num(z_num.detach().cpu().numpy(), x[:,0:k_num].cpu().numpy(), eps1= eps1)

        ## update and project categorical feature
        z_cat = z[:, k_num:] + 0.5 * torch.sign(grad[:, k_num:])
        z_cat = torch.clip(z_cat, min = 0, max = 15)

        ## concatenate
        z = torch.cat([torch.tensor(z_p_num, dtype = torch.float).cuda(), z_cat], dim = 1)
        z.requires_grad_()
        z.retain_grad()

    sample_x = get_gumbel_output3(z, number=m_num, tau=0.2, cat_list=cat_list)
    return sample_x
