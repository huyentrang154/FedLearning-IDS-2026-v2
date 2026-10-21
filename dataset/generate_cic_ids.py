import os
import sys
import random
import numpy as np
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ujson

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "cic_ids/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if niid:
        data_type = 'noniid'
    else:
        data_type = 'iid'
    # Setup directory for train/test data
    config_path = dir_path + data_type + str(num_clients) +partition+ "01/config.json"
    train_path = dir_path + data_type + str(num_clients) +partition+ "01/train/"
    test_path = dir_path + data_type + str(num_clients) +partition+ "01/test/"

    if check(config_path, train_path, test_path, num_clients, balance):
        return

    # load CIC_IDS data
    train_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Train-10percent.csv', sep=",", header=None)
    test_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Test-10percent.csv', sep=',', header=None)

    # drop nan value
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)

    # concatenate
    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)
    # y = data[85]
    # y.loc[y != "BENIGN"] = 1  # abnormal
    # y.loc[y == "BENIGN"] = 0  # normal
    # y = data[85].map(lambda x: 0 if x == "BENIGN" else 1)
    mapping = {
        'DoS GoldenEye': 0,
        'PortScan': 1,
        'SSH-Patator': 2,
        'DoS slowloris': 3,
        'Bot': 4,
        'Heartbleed': 5,
        'BENIGN': 6,
        'DDoS': 7,
        'DoS Slowhttptest': 8,
        'Infiltration': 9,
        'DoS Hulk': 10,
        'FTP-Patator': 11
    }
    y = data[85].map(mapping)
    droppedColumn = [0, 1, 2, 4, 7, 85]
    X = data
    for idx in droppedColumn:
        X = X.drop([idx], axis=1)

    X = X.to_numpy()

    X = np.array(X, dtype=np.float64)

    # normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = y.to_numpy()
    num_classes = len(set(y))
    print("Original Size: ", (X.shape, y.shape))
    print(f'Number of classes: {num_classes}')
    X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data, = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, balance)

# def generate_centralized_dataset(dir_path):
#     """
#     Sinh dataset cho Centralized Learning.
#     Không chia client.
#     """

#     train_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Train-10percent.csv', sep=",", header=None)
#     test_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Test-10percent.csv', sep=',', header=None)

#     # drop nan value
#     train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     train_data.dropna(inplace=True)
#     test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     test_data.dropna(inplace=True)

#     # concatenate
#     data = pd.concat([train_data, test_data])
#     data.reset_index(drop=True, inplace=True)
#     # y = data[85]
#     # y.loc[y != "BENIGN"] = 1  # abnormal
#     # y.loc[y == "BENIGN"] = 0  # normal
#     # y = data[85].map(lambda x: 0 if x == "BENIGN" else 1)
#     mapping = {
#         'DoS GoldenEye': 0,
#         'PortScan': 1,
#         'SSH-Patator': 2,
#         'DoS slowloris': 3,
#         'Bot': 4,
#         'Heartbleed': 5,
#         'BENIGN': 6,
#         'DDoS': 7,
#         'DoS Slowhttptest': 8,
#         'Infiltration': 9,
#         'DoS Hulk': 10,
#         'FTP-Patator': 11
#     }
#     y = data[85].map(mapping)
#     droppedColumn = [0, 1, 2, 4, 7, 85]
#     X = data
#     for idx in droppedColumn:
#         X = X.drop([idx], axis=1)

#     X = X.to_numpy()

#     X = np.array(X, dtype=np.float64)

#     # normalize data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

#     y = y.to_numpy()

#     print("====================================")
#     print("Centralized Dataset")
#     print("Samples :", len(y))
#     print("Features:", X.shape[1])
#     print("Classes :", len(np.unique(y)))
#     print("====================================")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         train_size=0.8,
#         random_state=42,
#         stratify=y,
#         shuffle=True
#     )

#     save_dir = os.path.join(dir_path, "centralized")
#     os.makedirs(save_dir, exist_ok=True)

#     np.savez_compressed(
#         os.path.join(save_dir, "train.npz"),
#         data={
#             "x": X_train,
#             "y": y_train
#         }
#     )

#     np.savez_compressed(
#         os.path.join(save_dir, "test.npz"),
#         data={
#             "x": X_test,
#             "y": y_test
#         }
#     )

#     config = {
#         "train_size": len(y_train),
#         "test_size": len(y_test),
#         "num_classes": len(np.unique(y)),
#         "input_dim": X.shape[1]
#     }

#     with open(os.path.join(save_dir, "config.json"), "w") as f:
#         ujson.dump(config, f)

#     print("Centralized dataset saved.")
#     print("Train:", len(y_train))
#     print("Test :", len(y_test))

# if __name__ == "__main__":
    
#     generate_centralized_dataset(
#         dir_path=dir_path
#     )

#     print("=" * 60)
#     print("Dataset generation completed!")
#     print("=" * 60)
    
# # if __name__ == "__main__":
# #     niid = True if sys.argv[1] == "noniid" else False
# #     balance = True if sys.argv[2] == "balance" else False
# #     partition = sys.argv[3] if sys.argv[3] != "-" else None

# #     generate_dataset(dir_path, num_clients, niid, balance, partition)

# #
# # cd ./dataset
# # # python generate_MNIST.py iid - - # for iid and unbalanced scenario
# # # python generate_MNIST.py iid balance - # for iid and balanced scenario
# # # python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
# # python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario
def generate_centralized_dataset(dir_path, data_type, num_clients):
    """
    Centralized:
        TRAIN: gộp train của 20 client thành 1 tập.

        TEST:
        KHÔNG gộp.
        Lưu riêng test của từng client:
            centralized/test/0.npz
            centralized/test/1.npz
            ...
            centralized/test/19.npz
    """

    data_root = os.path.join(
        dir_path,
        data_type + str(num_clients)
    )

    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")

    save_dir = os.path.join(data_root, "centralized")
    train_save_dir = os.path.join(save_dir, "train")
    test_save_dir = os.path.join(save_dir, "test")

    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    print("=" * 70)
    print("Creating Centralized Dataset")
    print("=" * 70)

    # ============================================================
    # 1. GỘP TRAIN DATA CỦA 20 CLIENT
    # ============================================================

    train_x_list = []
    train_y_list = []

    for client_id in range(num_clients):

        file_path = os.path.join(
            train_path,
            f"{client_id}.npz"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Không tìm thấy train file: {file_path}"
            )

        data = np.load(
            file_path,
            allow_pickle=True
        )

        client_data = data["data"].item()

        x = client_data["x"]
        y = client_data["y"]

        train_x_list.append(x)
        train_y_list.append(y)

        print(
            f"Train Client {client_id:2d}: "
            f"{len(y):8d} samples"
        )

    # GỘP TRAIN
    X_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    print("-" * 70)
    print(f"Total centralized train samples: {len(y_train)}")

    # ============================================================
    # 2. TEST DATA - KHÔNG GỘP
    # ============================================================

    test_x_list = []
    test_y_list = []

    for client_id in range(num_clients):

        file_path = os.path.join(
            test_path,
            f"{client_id}.npz"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Không tìm thấy test file: {file_path}"
            )

        data = np.load(
            file_path,
            allow_pickle=True
        )

        client_data = data["data"].item()

        x = client_data["x"]
        y = client_data["y"]

        # Lưu vào list
        test_x_list.append(x)
        test_y_list.append(y)

        # --------------------------------------------------------
        # LƯU TEST RIÊNG CHO TỪNG CLIENT
        # --------------------------------------------------------

        np.savez_compressed(
            os.path.join(
                test_save_dir,
                f"{client_id}.npz"
            ),
            data={
                "x": x,
                "y": y
            }
        )

        print(
            f"Test Client {client_id:2d}: "
            f"{len(y):8d} samples"
        )

    # ============================================================
    # 3. THÔNG TIN DATASET
    # ============================================================

    num_classes = len(
        np.unique(
            np.concatenate(
                [y_train] + test_y_list
            )
        )
    )

    input_dim = X_train.shape[1]

    print("=" * 70)
    print("Centralized Dataset")
    print("=" * 70)

    print(f"Number of clients : {num_clients}")
    print(f"Train samples     : {len(y_train)}")
    print(f"Train features    : {input_dim}")
    print(f"Number of classes : {num_classes}")

    print("\nTest samples per client:")
    for client_id in range(num_clients):
        print(
            f"  Client {client_id:2d}: "
            f"{len(test_y_list[client_id]):8d}"
        )

    print("=" * 70)

    # ============================================================
    # 4. SAVE CENTRALIZED TRAIN
    # ============================================================

    np.savez_compressed(
        os.path.join(
            train_save_dir,
            "all.npz"
        ),
        data={
            "x": X_train,
            "y": y_train
        }
    )

    print(
        f"Saved centralized train: "
        f"{os.path.join(train_save_dir, 'all.npz')}"
    )

    # ============================================================
    # 5. SAVE CONFIG
    # ============================================================

    config = {
        "num_clients": num_clients,
        "train_size": len(y_train),
        "test_sizes": [
            len(y)
            for y in test_y_list
        ],
        "num_classes": num_classes,
        "input_dim": input_dim
    }

    with open(
        os.path.join(save_dir, "config.json"),
        "w"
    ) as f:
        ujson.dump(config, f)

    print("=" * 70)
    print("Centralized dataset saved.")
    print("=" * 70)
# if __name__ == "__main__":
    
#     # generate_centralized_dataset(
#     #     dir_path=dir_path
#     # )

#     # print("=" * 60)
#     # print("Dataset generation completed!")
#     # print("=" * 60)
#     niid = True if sys.argv[1] == "noniid" else False
#     balance = True if sys.argv[2] == "balance" else False
#     partition = sys.argv[3] if sys.argv[3] != "-" else None

#     generate_dataset(dir_path, num_clients, niid, balance, partition)
if __name__ == "__main__":

    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # ============================================================
    # 1. Generate FL dataset
    # ============================================================

    # generate_dataset(
    #     dir_path,
    #     num_clients,
    #     niid,
    #     balance,
    #     partition
    # )

    # ============================================================
    # 2. Generate Centralized dataset
    #    bằng cách gộp 20 client
    # ============================================================

    data_type = "noniid" if niid else "iid"

    generate_centralized_dataset(
        dir_path=dir_path,
        data_type=data_type,
        num_clients=num_clients
    )

    print("=" * 70)
    print("Dataset generation completed!")
    print("=" * 70)
    