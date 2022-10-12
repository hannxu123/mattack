
import numpy as np
from preprocess2 import get_criteo_data
import torch
import torch.optim as optim
from util2 import *

def main():
    X_train, X_test, y_train, y_test, num_names, cat_names, final_names = get_criteo_data(seed = 2000)
    model = Net(X_train.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, X_train, y_train, X_test, y_test, optimizer)

if __name__ == "__main__":
    main()

