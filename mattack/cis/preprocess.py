
from util import *
from sklearn.preprocessing import LabelEncoder

def main():
    RAW_DATA_PATH = "./dataset/"
    print("Loading data....")
    train_merge = load_and_merge(RAW_DATA_PATH, 'train')
    print("Finish Loading data....")
    train_merge = reduce_mem_usage(train_merge)
    print("Reduction of memory success")

    print(f"Merged training set shape: {train_merge.shape}")
    train_merge.drop(columns = ['id_12', 'TransactionID_x', 'TransactionID_y'], inplace=True)

    for col in train_merge.columns:
        if train_merge[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(train_merge[col].values))
            train_merge[col] = le.transform(list(train_merge[col].values))

    train_merge.to_csv('./dataset/processed_cis.csv', index=False)


if __name__ == "__main__":
    main()

