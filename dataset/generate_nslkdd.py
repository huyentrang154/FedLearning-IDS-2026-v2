# import os
# import sys
# import random
# import numpy as np
# import pandas as pd
# from utils.dataset_utils import check, separate_data, split_data, save_file
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import ujson

# random.seed(1)
# np.random.seed(1)
# num_clients = 20
# dir_path = "nslkdd/"


# # Allocate data to users
# def generate_dataset(dir_path, num_clients, niid, balance, partition):
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)

#     if niid:
#         data_type = 'noniid'
#     else:
#         data_type = 'iid'
#     # Setup directory for train/test data
#     config_path = dir_path + data_type + str(num_clients) +partition+ "/config.json"
#     train_path = dir_path + data_type + str(num_clients) +partition+ "/train/"
#     test_path = dir_path + data_type + str(num_clients) +partition+ "/test/"

#     if check(config_path, train_path, test_path, num_clients, balance):
#         return

#     # load NSLKDD data
#     train_data = np.genfromtxt(f'{dir_path}/rawdata/NSLKDD_Train.csv', delimiter=",")
#     test_data = np.genfromtxt(f'{dir_path}/rawdata/NSLKDD_Test.csv', delimiter=",")
#     data = np.concatenate((train_data, test_data), axis=0)
#     print('Data size:', data.shape)

#     # split x, y
#     X = data[:, :-1]
#     y = data[:, -1]

#     X = np.array(X, dtype=np.float64)

#     # normalize data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

#     # reassign labels
#     # y[y != 0] = 1  # abnormal

#     num_classes = len(set(y))
#     print("Original Size: ", (X.shape, y.shape))
#     print(f'Number of classes: {num_classes}')
#     # X, y, statistic = separate_data(X, y, num_clients, num_classes, balance)
#     X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
#     train_data, test_data = split_data(X, y)
#     save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
#               statistic, balance)

# def generate_centralized_dataset(dir_path):
#     """
#     Sinh dataset cho Centralized Learning.
#     Không chia client.
#     """

#     train_data = np.genfromtxt(
#         f"{dir_path}/rawdata/NSLKDD_Train.csv",
#         delimiter=","
#     )

#     test_data = np.genfromtxt(
#         f"{dir_path}/rawdata/NSLKDD_Test.csv",
#         delimiter=","
#     )

#     data = np.concatenate((train_data, test_data), axis=0)

#     X = data[:, :-1]
#     y = data[:, -1]

#     X = np.array(X, dtype=np.float32)

#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

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
#     niid = True if sys.argv[1] == "noniid" else False
#     balance = True if sys.argv[2] == "balance" else False
#     partition = sys.argv[3] if sys.argv[3] != "-" else None

#     generate_dataset(dir_path, num_clients, niid, balance, partition)

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
dir_path = "nslkdd/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if niid:
        data_type = 'noniid'
    else:
        data_type = 'iid'
    # Setup directory for train/test data
    config_path = dir_path + data_type + str(num_clients) +partition+ "/config.json"
    train_path = dir_path + data_type + str(num_clients) +partition+ "/train/"
    test_path = dir_path + data_type + str(num_clients) +partition+ "/test/"

    if check(config_path, train_path, test_path, num_clients, balance):
        return

    # load NSLKDD data
    train_data = np.genfromtxt(f'{dir_path}/rawdata/NSLKDD_Train.csv', delimiter=",")
    test_data = np.genfromtxt(f'{dir_path}/rawdata/NSLKDD_Test.csv', delimiter=",")
    data = np.concatenate((train_data, test_data), axis=0)
    print('Data size:', data.shape)

    # split x, y
    X = data[:, :-1]
    y = data[:, -1]

    X = np.array(X, dtype=np.float64)

    # normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # reassign labels
    # y[y != 0] = 1  # abnormal

    num_classes = len(set(y))
    print("Original Size: ", (X.shape, y.shape))
    print(f'Number of classes: {num_classes}')
    # X, y, statistic = separate_data(X, y, num_clients, num_classes, balance)
    X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, balance)

# def generate_centralized_dataset(dir_path):
#     """
#     Sinh dataset cho Centralized Learning.
#     Không chia client.
#     """

#     train_data = np.genfromtxt(
#         f"{dir_path}/rawdata/NSLKDD_Train.csv",
#         delimiter=","
#     )

#     test_data = np.genfromtxt(
#         f"{dir_path}/rawdata/NSLKDD_Test.csv",
#         delimiter=","
#     )

#     data = np.concatenate((train_data, test_data), axis=0)

#     X = data[:, :-1]
#     y = data[:, -1]

#     X = np.array(X, dtype=np.float32)

#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

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

# def generate_centralized_dataset(dir_path, data_type, num_clients):
#     """
#     Tạo dataset Centralized bằng cách gộp dữ liệu train/test
#     của toàn bộ num_clients client.

#     Ví dụ:
#         noniid20/train/0.npz ... 19.npz
#         noniid20/test/0.npz  ... 19.npz

#     Centralized:
#         noniid20/centralized/train.npz
#         noniid20/centralized/test.npz
#     """

#     data_root = os.path.join(
#         dir_path,
#         data_type + str(num_clients)
#     )

#     train_path = os.path.join(data_root, "train")
#     test_path = os.path.join(data_root, "test")

#     save_dir = os.path.join(data_root, "centralized")
#     os.makedirs(save_dir, exist_ok=True)

#     print("=" * 70)
#     print("Creating Centralized Dataset")
#     print("=" * 70)

#     # ============================================================
#     # 1. GỘP TRAIN DATA CỦA 20 CLIENT
#     # ============================================================

#     train_x_list = []
#     train_y_list = []

#     for client_id in range(num_clients):

#         file_path = os.path.join(
#             train_path,
#             f"{client_id}.npz"
#         )

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(
#                 f"Không tìm thấy train file: {file_path}"
#             )

#         data = np.load(
#             file_path,
#             allow_pickle=True
#         )

#         client_data = data["data"].item()

#         x = client_data["x"]
#         y = client_data["y"]

#         train_x_list.append(x)
#         train_y_list.append(y)

#         print(
#             f"Train Client {client_id:2d}: "
#             f"{len(y):8d} samples"
#         )

#     X_train = np.concatenate(train_x_list, axis=0)
#     y_train = np.concatenate(train_y_list, axis=0)

#     # ============================================================
#     # 2. GỘP TEST DATA CỦA 20 CLIENT
#     # ============================================================

#     test_x_list = []
#     test_y_list = []

#     for client_id in range(num_clients):

#         file_path = os.path.join(
#             test_path,
#             f"{client_id}.npz"
#         )

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(
#                 f"Không tìm thấy test file: {file_path}"
#             )

#         data = np.load(
#             file_path,
#             allow_pickle=True
#         )

#         client_data = data["data"].item()

#         x = client_data["x"]
#         y = client_data["y"]

#         test_x_list.append(x)
#         test_y_list.append(y)

#         print(
#             f"Test  Client {client_id:2d}: "
#             f"{len(y):8d} samples"
#         )

#     X_test = np.concatenate(test_x_list, axis=0)
#     y_test = np.concatenate(test_y_list, axis=0)

#     # ============================================================
#     # 3. THÔNG TIN DATASET
#     # ============================================================

#     num_classes = len(
#         np.unique(
#             np.concatenate([y_train, y_test])
#         )
#     )

#     input_dim = X_train.shape[1]

#     print("=" * 70)
#     print("Centralized Dataset")
#     print("=" * 70)
#     print(f"Number of clients : {num_clients}")
#     print(f"Train samples     : {len(y_train)}")
#     print(f"Test samples      : {len(y_test)}")
#     print(f"Train features    : {X_train.shape[1]}")
#     print(f"Test features     : {X_test.shape[1]}")
#     print(f"Number of classes : {num_classes}")
#     print("=" * 70)

#     # ============================================================
#     # 4. SAVE CENTRALIZED TRAIN
#     # ============================================================

#     np.savez_compressed(
#         os.path.join(save_dir, "train.npz"),
#         data={
#             "x": X_train,
#             "y": y_train
#         }
#     )

#     # ============================================================
#     # 5. SAVE CENTRALIZED TEST
#     # ============================================================

#     np.savez_compressed(
#         os.path.join(save_dir, "test.npz"),
#         data={
#             "x": X_test,
#             "y": y_test
#         }
#     )

#     # ============================================================
#     # 6. SAVE CONFIG
#     # ============================================================

#     config = {
#         "num_clients": num_clients,
#         "train_size": int(len(y_train)),
#         "test_size": int(len(y_test)),
#         "num_classes": int(num_classes),
#         "input_dim": int(input_dim)
#     }

#     with open(
#         os.path.join(save_dir, "config.json"),
#         "w"
#     ) as f:
#         ujson.dump(config, f, indent=4)

#     print("Centralized dataset saved successfully.")
#     print(f"Train: {len(y_train)}")
#     print(f"Test : {len(y_test)}")
#     print(
#         f"Location: {save_dir}"
#     )

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
    