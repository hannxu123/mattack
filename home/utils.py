import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from augmentations import ClassificationSMOTE
from sklearn.decomposition import TruncatedSVD

class Net_random(nn.Module):
    def __init__(self, input, hidden_size):
      super(Net_random, self).__init__()
      self.fc1 = nn.Linear(input, hidden_size)
      self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x

def svd_mixed(X_train, cat_length, dataset = 'home'):
    X2 = X_train[:,cat_length:]
    Z_n = X_train[:,0:cat_length]
    N = np.sum(X2)

    c = np.expand_dims(np.mean(X2, axis=0), 0)
    rc_t = c
    Z_c = (X2 - rc_t)
    Z = np.concatenate([Z_n, Z_c], axis= 1)

    random_idx = np.random.choice(Z.shape[0], int(0.8 * Z.shape[0]), replace= False)
    Z = Z[random_idx]

    svd = TruncatedSVD(n_components = 20)
    svd.fit(Z)
    s = svd.singular_values_
    Vh = svd.components_

    s = s / np.sqrt(N)
    np.save('./svd_logs/s_'  + dataset, s)
    np.save('./svd_logs/Vh_' + dataset, Vh)

    return c, s, Vh


class Net2(nn.Module):
    def __init__(self, input):
      super(Net2, self).__init__()
      self.fc1 = nn.Linear(input, 32)
      self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x




def train2(model, X_train, y_train, optimizer):

    ## loss function
    criterion = nn.BCELoss()

    for i in range(5000):
        pos_ind = np.random.choice(np.where(y_train == 1)[0], 128, replace=False)
        neg_ind = np.random.choice(np.where(y_train == 0)[0], 128, replace=False)
        train_idx = np.concatenate([pos_ind, neg_ind])

        data = torch.tensor(X_train[train_idx], dtype = torch.float).cuda()
        target = torch.tensor(y_train[train_idx], dtype = torch.float).cuda()

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(torch.sigmoid(output).flatten(), target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def train(model, X_train, y_train, X_test, y_test, optimizer):

    ## loss function
    criterion = nn.BCELoss()

    for i in range(10000):
        pos_ind = np.random.choice(np.where(y_train == 1)[0], 128, replace=False)
        neg_ind = np.random.choice(np.where(y_train == 0)[0], 128, replace=False)
        train_idx = np.concatenate([pos_ind, neg_ind])

        data = torch.tensor(X_train[train_idx], dtype = torch.float).cuda()
        target = torch.tensor(y_train[train_idx], dtype = torch.float).cuda()

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(torch.sigmoid(output).flatten(), target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i % 2000 == 0) & (i > 0) :
            print('..............Step ' + str(i))
            test(model, X_test, y_test)

    torch.save(model.state_dict(), './models/mlp1.pt')


def test(model, X_test, y_test):

    all_label = []
    all_pred = []

    for i in range(200):
        pos_ind = np.random.choice(np.where(y_test == 1)[0], 64, replace=False)
        neg_ind = np.random.choice(np.where(y_test == 0)[0], 64, replace=False)
        train_idx = np.concatenate([pos_ind, neg_ind])

        data = torch.tensor(X_test[train_idx], dtype = torch.float).cuda()
        target = torch.tensor(y_test[train_idx], dtype = torch.float).cuda()

        pred = (model(data) > 0.0)
        all_pred.append(pred.cpu().numpy())
        all_label.append(target.cpu().numpy())

    all_pred = np.concatenate(all_pred).flatten()
    all_label = np.concatenate(all_label).flatten()

    acc = np.sum(all_pred == all_label) / all_label.shape[0]
    acc1 = np.sum((all_pred == all_label) & (all_label == 1)) / np.sum(all_label == 1)
    acc0 = np.sum((all_pred == all_label) & (all_label == 0)) / np.sum(all_label == 0)

    print('Test Result')
    print(acc ,acc1, acc0)

def check_attack(sample_x, data, num_names):
    diff = sample_x - data
    l1_dist = torch.sum(torch.abs(diff[0,0:len(num_names)])).item()

    cat_changed = 0
    org_list = (torch.where(diff[0] == -1))[0]
    for k in range(org_list.shape[0]):
        cat_changed = cat_changed + 1

    return l1_dist, cat_changed

def get_ce_loss(z, x, cat_list = None):
    sum_ce = 0
    for j in range(len(cat_list) - 1):
        z2 = z[:, cat_list[j]:cat_list[j + 1]]
        sum_ce = sum_ce + (-torch.log(torch.sum(F.softmax(z2 + 1e-4) * x[:, cat_list[j]:cat_list[j + 1]])))
    return sum_ce

def get_gumbel_output2(z, cat_list = None, number=30, tau = 0.2):
    data_new = z[:, 0: cat_list[0]].repeat(number, 1)
    ## stack the categorical probability
    for j in range(len(cat_list) - 1):
        z2 = (z[:, cat_list[j]:cat_list[j + 1]]).repeat(number, 1)
        samples = F.gumbel_softmax(z2 + 1e-3, tau= tau, hard= True)
        data_new = torch.cat([data_new, samples], dim=1)
    return data_new

def get_gumbel_output3(z, cat_list = None, number=30, tau = 0.2):
    data_new = z[:, 0: cat_list[0]].repeat(number, 1)
    ## stack the categorical probability
    for j in range(len(cat_list) - 1):
        z2 = (z[:, cat_list[j]:cat_list[j + 1]]).repeat(number, 1)
        samples = F.gumbel_softmax(z2 + 1e-3, tau= tau, hard=True)
        data_new = torch.cat([data_new, samples], dim=1)
    return data_new
