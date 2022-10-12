import os
import gc
import numpy as np
import pandas as pd
from util import *
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from time import time
import datetime
from sklearn.model_selection import TimeSeriesSplit

def base_model(X, y, N_SPLITS):
    """
    Base Model for the dataset. Return the feature importance dataframe
    to identify important features .

    :param X:  dataframe. Containg the training data
    :param Y: dataframe Containg the fraud label
    :param N_SPLITS:  int. the number of  folds
    :return: feature_importances, dataframe that contain feature importance for each fold
    """
    folds = TimeSeriesSplit(n_splits=N_SPLITS)
    model = lgb.LGBMClassifier(random_state=50)
    hyperparameters = model.get_params()
    hyperparameters['metric'] = 'auc'

    aucs = list()
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns

    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
        start_time = time()
        print(f"Training on fold {fold + 1}")

        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
        clf = lgb.train(hyperparameters, trn_data, 10000, valid_sets=[trn_data, val_data],
                        verbose_eval=1000, early_stopping_rounds=500)

        feature_importances[f'fold_{fold + 1}'] = clf.feature_importance()
        aucs.append(clf.best_score['valid_1']['auc'])

        fold_end_time = datetime.timedelta(seconds=time() - start_time)

        print(f'Fold {fold + 1} finished in {str(fold_end_time)}')

    print('-' * 30)
    print('Train finished')
    print(f'Mean AUC: {np.mean(aucs)}')
    print('-' * 30)

    return feature_importances


def main():
    train_merge = pd.read_csv('./dataset/processed_cis.csv')
    X_train = train_merge.sort_values('TransactionDT').drop(['isFraud'], axis=1)
    y_fraud = train_merge.sort_values('TransactionDT')['isFraud']

    fold = 5
    feature_importance = base_model(X_train, y_fraud, fold)
    feature_importance['average'] = feature_importance[[f'fold_{fold + 1}' for fold in range(fold)]].mean(
        axis=1)
    feature_importance.to_csv('feature_importances.csv', index = False)

    feature_importance['mean'] = feature_importance.iloc[:, 1:fold].mean(axis=1)


    important_vec = np.array(feature_importance['mean'])
    impt_features = []

    for i in range(important_vec.shape[0]):
        name = (feature_importance.iloc[i,0])
        bb = (feature_importance.iloc[i, fold + 1])
        if ('V' not in name):
            impt_features.append(name)
        else:
            if bb > 5:
                impt_features.append(name)

    print((impt_features))

if __name__ == "__main__":
    main()

