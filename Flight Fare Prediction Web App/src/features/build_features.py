import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(train_data_path: str, test_data_path: str, save_file_path: str):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        train_data, test_data, test_size=0.2
    )
    X_train.to_csv(f"{save_file_path}/X_train.csv", index=False)
    X_test.to_csv("{save_file_path}/X_test.csv", index=False)
    y_train.to_csv("{save_file_path}/y_train.csv", index=False)
    y_test.to_csv("{save_file_path}/y_test.csv", index=False)
