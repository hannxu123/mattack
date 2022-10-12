import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
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
    prob.solve()

    return x1.value

def l1_attack(model, x, y,
               k_num,
               num_steps = 15,
               eps1 = 0.5):

    ## attack setups
    criterion = nn.BCELoss()
    z = torch.clone(x)
    z.requires_grad_()

    ## do pgd on numerical feature
    for j in range(num_steps):
        output = model(z)
        loss = criterion(torch.sigmoid(output).flatten(), y.flatten())
        grad = torch.autograd.grad(loss, z)[0]

        ## update and project numerical feature
        z_num = z[:, 0:k_num] + 0.05 * grad[:, 0:k_num]
        try:
            z_p_num = linear_proj_num(z_num.detach().cpu().numpy(), x[:,0:k_num].cpu().numpy(), eps1= eps1)
        except:
            z_p_num = linear_proj_num(z_num.detach().cpu().numpy() + 0.001, x[:,0:k_num].cpu().numpy(), eps1= eps1)

        ## concatenate
        z = torch.cat([torch.tensor(z_p_num, dtype = torch.float).cuda(), z[:, k_num:]], dim = 1)
        z.requires_grad_()
        z.retain_grad()

    return z

def grad_attack(model, x, y,
                cat_features,
                all_features,
                k_num = 108,
                num_steps = 20,
                eps1 = 0.5,
                eps2 = 3, s=None, Vh=None, m_weight = 2):

    ## attack setups
    s_inv = np.diag(1 / s)
    M = torch.tensor((Vh.T).dot(s_inv), dtype = torch.float).cuda()
    criterion = nn.BCELoss()
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
    loss_vec = torch.zeros(len(cat_list)-1)

    ############################################## find target set
    with torch.no_grad():
        for i in range(len(cat_list)-1):
            z = torch.clone(x[:, cat_list[i]: cat_list[i + 1]])
            loss_v = torch.zeros(z.shape[1])
            for j in range(z.shape[1]):
                if z[:, j] < 1:
                    z_new = torch.zeros(z.shape).cuda()
                    z_new[:, j] = 1
                    x1 = torch.clone(x)
                    x1[:, cat_list[i]: cat_list[i + 1]] = z_new
                    output = model(x1)
                    loss = criterion(torch.sigmoid(output).flatten(), y.flatten())
                    loss = loss - m_weight * torch.norm(torch.matmul(x1, M) - torch.matmul(x, M))
                    loss_v[j] = loss_v[j] + loss
            loss_vec[i] = loss_vec[i] + torch.max(loss_v)

    ################################################ target attack set
    v, attack_set = torch.topk(torch.abs(loss_vec), k= eps2)
    attack_list = []

    ## doing numerical attack using L1 PGD
    z = l1_attack(model, x, y, k_num, num_steps, eps1)

    with torch.no_grad():
        for i in attack_set:
            attack_list.append(np.arange(cat_list[i], cat_list[i + 1]))
            z[:, cat_list[i]:cat_list[i + 1]] = 0 * z[:, cat_list[i]:cat_list[i + 1]]
        best = -1e3
        best_set = []
        if eps2 == 1:
            aa_list = product(attack_list[0])
        if eps2 == 2:
            aa_list = product(attack_list[0], attack_list[1])
        if eps2 == 3:
            aa_list = product(attack_list[0], attack_list[1], attack_list[2])
        if eps2 == 4:
            aa_list = product(attack_list[0], attack_list[1], attack_list[2], attack_list[3])
        if eps2 == 5:
            aa_list = product(attack_list[0], attack_list[1], attack_list[2], attack_list[3], attack_list[4])

        for i in aa_list:
            z1 = torch.clone(z)
            z1[:, i] = z1[:, i] + 1

            output = model(z1)
            loss = criterion(torch.sigmoid(output).flatten(), y.flatten())
            loss = loss - m_weight * torch.norm(torch.matmul(z1, M) - torch.matmul(x, M))

            if loss > best:
                best = loss
                best_set = i
        z[:, best_set] = z[:, best_set] + 1
    return z

