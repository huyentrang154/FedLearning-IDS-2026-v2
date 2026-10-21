import os
import sys
import random
import numpy as np
import pandas as pd
import ujson

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.dataset_utils import (
    check,
    separate_data,
    split_data,
    save_file
)

#########################################################
# Configuration
#########################################################

random.seed(1)
np.random.seed(1)

num_clients = 20

dir_path = "toniot/"

#########################################################
# Preprocess TON-IoT
#########################################################

def preprocess_toniot(dir_path):
    """
    Đọc và tiền xử lý TON-IoT Network Dataset.

    Output
    ------
    X : np.ndarray
        Feature matrix đã encode + normalize

    y : np.ndarray
        Label (0/1)
    """

    csv_path = os.path.join(
        dir_path,
        "rawdata",
        "train_test_network.csv"
    )

    print("=" * 60)
    print("Loading TON-IoT dataset ...")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print("Original shape :", df.shape)

    ####################################################
    # Replace missing value '-'
    ####################################################

    df = df.replace("-", "Unknown")
    df = df.fillna("Unknown")

    ####################################################
    # Multi-class label
    ####################################################

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["type"])

    print("\nClass Mapping")
    print("-" * 40)

    for i, c in enumerate(label_encoder.classes_):
        print(f"{i:2d} -> {c}")

    ####################################################
    # Features
    ####################################################

    X = df.drop(columns=["label", "type"])

    ####################################################
    # Encode categorical columns
    ####################################################

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns

    print("Categorical columns:")
    print(list(categorical_cols))

    for col in categorical_cols:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

    ####################################################
    # Convert numeric
    ####################################################

    X = X.astype(np.float32)

    ####################################################
    # Normalize
    ####################################################

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    ####################################################
    # Dataset information
    ####################################################

    print("=" * 60)
    print("TON-IoT preprocessing completed")
    print("=" * 60)
    print("Samples      :", len(y))
    print("Features     :", X.shape[1])
    print("Number of classes :", len(label_encoder.classes_))

    print("\nClass distribution")

    for idx, cls in enumerate(label_encoder.classes_):
        print(f"{cls:20s}: {(y == idx).sum()}")
    print("=" * 60)

    return X, y
#########################################################
# Generate Federated Dataset
#########################################################

def generate_dataset(
        dir_path,
        num_clients,
        niid,
        balance,
        partition
):
    """
    Sinh dataset cho Federated Learning.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    ####################################################
    # Dataset type
    ####################################################

    if niid:
        data_type = "noniid"
    else:
        data_type = "iid"

    save_dir = (
        dir_path
        + data_type
        + str(num_clients)
        + partition
    )

    config_path = os.path.join(
        save_dir,
        "config.json"
    )

    train_path = os.path.join(
        save_dir,
        "train/"
    )

    test_path = os.path.join(
        save_dir,
        "test/"
    )

    ####################################################
    # Check existed dataset
    ####################################################

    if check(
            config_path,
            train_path,
            test_path,
            num_clients,
            balance
    ):
        return

    ####################################################
    # Preprocess TON-IoT
    ####################################################

    X, y = preprocess_toniot(dir_path)

    num_classes = len(np.unique(y))

    print("=" * 60)
    print("Generate Federated Dataset")
    print("=" * 60)
    print("Samples      :", len(y))
    print("Features     :", X.shape[1])
    print("Classes      :", num_classes)
    print("Clients      :", num_clients)
    print("Partition    :", partition)
    print("Non-IID      :", niid)
    print("=" * 60)

    ####################################################
    # Partition dataset
    ####################################################

    X,y,statistic = separate_data(
        (X, y),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=2
    )

    ####################################################
    # Split train/test
    ####################################################

    train_data, test_data = split_data(
        X,
        y
    )

    ####################################################
    # Save dataset
    ####################################################

    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition
    )

    print("=" * 60)
    print("TON-IoT Federated dataset generated successfully.")
    print("=" * 60)

#########################################################
# Generate Centralized Dataset
#########################################################

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
    