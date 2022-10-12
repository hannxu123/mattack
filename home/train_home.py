
import numpy as np
import pandas as pd 
## train model
from utils import *
import torch.optim as optim
from home_preprocess import get_home_data

# one-hot encoding of categorical variables
X_train, X_test, y_train, y_test, num_names, cat_names, final_names = get_home_data(seed=2000)

model = Net2(X_train.shape[1]).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train(model, X_train, y_train, X_test, y_test, optimizer)