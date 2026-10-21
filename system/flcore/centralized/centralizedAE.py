# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# from utils.data_utils import read_centralized_data
# from torch.utils.data import DataLoader
# import os
# from sklearn import metrics
# torch.cuda.empty_cache()
# class CentralizeAE(object):
#     def __init__(self, args, times):
#         self.model = copy.deepcopy(args.model).to(args.device)
#         self.dataset = args.dataset
#         self.device = args.device
#         self.input_channels = args.input_channels
#         self.image_size = args.image_size

#         self.batch_size = args.batch_size
#         self.learning_rate = args.local_learning_rate
#         self.local_steps = args.local_steps


#         # Loss & optimizer

#         # Data loading
#         self.data_type = args.data_type
#         full_train_data = read_centralized_data(
#             self.dataset, data_type='train'
#         )

#         # Split train/val (90/10)
#         self.train_data, self.val_data = self._split_train_val(full_train_data, val_ratio=0.1)

#         self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=True)
#         self.val_loader = DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False)
#         self.test_loader = self.load_test_data()

#         self.rs_train_loss = []
#         self.rs_test_acc = []
#         self.rs_test_f1 = []
#         self.rs_test_precision = []
#         self.rs_test_recall = []

#         # Early stopping configs
#         self.patience = getattr(args, "patience", 5)
#         self.min_delta = getattr(args, "min_delta", 1e-4)
#         self.early_stop_metric = getattr(args, "early_stop_metric", "f1")

#         self.budget = []

#         self.result_path = os.path.join(
#             '..',
#             'results',
#             self.dataset,
#             self.data_type,
#             'CentralizedAE',
#             self.exp_name,
#             f"seed_{self.seed}"
#         )
#         os.makedirs(self.result_path, exist_ok=True)

#         self.log_file = os.path.join(self.result_path, "tracking.log")
    
#     def _process_batch(self, x, y):
#         """Xử lý shape và device cho từng batch dữ liệu."""
#         x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#         y = y.to(self.device).long()

#         # Nếu là model CNN/CNNAE và dữ liệu đang ở dạng vector phẳng (như kết quả từ script của bạn)
#         if self.input_channels>1 and len(x.shape) == 2 and self.image_size > 0:
#             x = x.view(-1, self.input_channels, self.image_size, self.image_size)
            
#         return x, y



#     # ---------------- Training (Giữ nguyên logic của bạn) ----------------
#     def train(self):
#         self.model.train()
#         best_state = None
#         no_improve = 0
#         best_score = -float("inf") if self.early_stop_metric == "f1" else float("inf")

#         for epoch in range(self.local_steps):
#             for x, y in self.train_loader:
#                 x, y = self._process_batch(x, y)
#                 output = self.model(x)
#                 if isinstance(output, tuple): output = output[0]
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#             val_stats = self.evaluate_on_val()
#             val_loss, val_f1 = val_stats["loss"], val_stats["f1"]

#             improved = False
#             if self.early_stop_metric == "f1":
#                 if val_f1 > best_score + self.min_delta:
#                     improved = True
#                     best_score = val_f1
#             else:
#                 if val_loss < best_score - self.min_delta:
#                     improved = True
#                     best_score = val_loss

#             if improved:
#                 # best_state = copy.deepcopy(self.model.state_dict())
#                 best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
#                 no_improve = 0
#             else:
#                 no_improve += 1
#                 if no_improve >= self.patience: break
#             torch.cuda.empty_cache()
#         self.model.to("cpu")
#         torch.cuda.empty_cache()

#         if best_state is not None:
#             # self.model.load_state_dict(best_state)
#             self.model.load_state_dict(best_state)
#             self.model.to(self.device)
#         print("\nAlgorithm:", self.algo, "Dataset:", self.dataset)
#         print("\nBest performance metrics")

#         print("\tacc:", max(self.rs_test_acc))
#         print("\tf1:", max(self.rs_test_f1))
#         print("\trecall:", max(self.rs_test_recall))
#         print("\tprecision:", max(self.rs_test_precision))

#         avg_time = np.mean(self.budget)

#         print("\tTime:", avg_time)

#         summary_lines = [
#             f"\nAlgorithm: {self.algo} | Dataset: {self.dataset}\n",
#             "\nBest performance metrics\n",
#             f"\tacc: {max(self.rs_test_acc):.4f}\n",
#             f"\tf1: {max(self.rs_test_f1):.4f}\n",
#             f"\trecall: {max(self.rs_test_recall):.4f}\n",
#             f"\tprecision: {max(self.rs_test_precision):.4f}\n",
#             f"\tAvg Time/Epoch: {avg_time:.4f} sec\n"
#         ]

#         for line in summary_lines:
#             print(line.strip())
#             with open(self.log_file, "a") as f:
#                 f.write(line)

#         self.save_results()


#     def evaluate_on_val(self):
#         self.model.eval()
#         losses = []
        
#         all_preds = []
#         all_labels = []

#         with torch.no_grad():
#             for x, y in self.val_loader:
#                 x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y = y.to(self.device).long()

#                 output = self.model(x)
#                 if isinstance(output, tuple):
#                     output = output[0]

#                 loss = self.loss(output, y)
#                 losses.append(loss.item())

#                 preds = torch.argmax(output, dim=1)

#                 # chuyển CPU ngay
#                 all_preds.append(preds.cpu())
#                 all_labels.append(y.cpu())

#                 # 🔥 giải phóng GPU
#                 del x, y, output, preds, loss

#             torch.cuda.empty_cache()

#         y_pred = torch.cat(all_preds).numpy()
#         y_true = torch.cat(all_labels).numpy()

#         f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)

#         return {
#             "loss": float(np.mean(losses)),
#             "f1": f1
#         }

#     def load_test_data(self, batch_size=1):
#         test_data = read_centralized_data(self.dataset, data_type='test')
#         return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

#     def test_metrics(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()

#         y_true = []
#         y_predict = []

#         with torch.no_grad():
#             for x, y in self.test_loader:
#                 x, y = self._process_batch(x, y)
#                 output = model(x)
#                 if isinstance(output, (list, tuple)):
#                     output = output[0]
#                 y_predict.append(torch.argmax(output, dim=1).detach().cpu().numpy())
#                 y_true.append(y.detach().cpu().numpy())

#         y_predict = np.concatenate(y_predict, axis=0)
#         y_true = np.concatenate(y_true, axis=0)

#         acc = metrics.accuracy_score(y_true, y_predict)
#         precision = metrics.precision_score(y_true, y_predict, average='macro')
#         recall = metrics.recall_score(y_true, y_predict, average='macro')
#         f1 = metrics.f1_score(y_true, y_predict, average='macro')
#         # auc = metrics.roc_auc_score(y_true, y_predict, average='micro')
#         test_num = len(y_predict)
#         # import pdb
#         # pdb.set_trace()

#         # return acc, precision, recall, f1, auc, test_num
#         return acc, precision, recall, f1, test_num
    
#     def train_metrics(self, model=None):
#         if model is None: model = self.model
#         model.eval()
#         list_loss = []
#         with torch.no_grad():
#             for x, y in self.train_loader:
#                 x, y = self._process_batch(x, y)
#                 output = model(x)
#                 if isinstance(output, tuple): output = output[0]
#                 loss = self.loss(output, y.long())
#                 list_loss.append(loss.item())
#         return list_loss
    
#     def evaluate(self):
#         # evaluate train
#         stats_train = self.train_metrics()
#         train_loss = sum(stats_train[1]) * 1.0 / sum(stats_train[0])  # avg train loss for all clients
#         self.rs_train_loss.append(train_loss)

#         # evaluate test, avg metrics for all clients
#         stats_test = self.test_metrics()
#         # test_acc = sum(stats_test[0]) / stats_test[5]
#         # test_auc = sum(stats_test[1]) / stats_test[5]
#         # test_f1 = sum(stats_test[2]) / stats_test[5]
#         # test_recall = sum(stats_test[3]) / stats_test[5]
#         # test_precision = sum(stats_test[4]) / stats_test[5]

#         test_acc_normal = sum(stats_test[0]) / stats_test[4]
#         test_f1_normal = sum(stats_test[1]) / stats_test[4]
#         test_recall_normal = sum(stats_test[2]) / stats_test[4]
#         test_precision_normal = sum(stats_test[3]) / stats_test[4]



#         self.rs_test_acc_normal.append(test_acc_normal)
#         # self.rs_test_auc.append(test_auc)
#         self.rs_test_f1_normal.append(test_f1_normal)
#         self.rs_test_precision_normal.append(test_precision_normal)
#         self.rs_test_recall_normal.append(test_recall_normal)


#         print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
#         print("Overall Performance Metrics")
#         print("\ttrain loss avg: {:.4f}".format(train_loss))


#         with open(self.log_file, 'a') as f:
#             f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
#             f.write("Overall Performance Metrics\n")
#             f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))



#     def train_metrics(self):
#         num_samples, losses = [], []
#         print("Train Loss")
#         for c in self.selected_clients:
#             list_loss = c.train_metrics()
#             loss = sum(list_loss) / len(list_loss)
#             self.train_losses[c.id].append(loss)
#             print(f'\tClient {c.id}: Train loss: {round(loss, 2)}')
#             num_samples.append(len(list_loss))
#             losses.extend(list_loss)
#         return num_samples, losses

#     def save_results(self):
#         # Đã cập nhật để lưu thêm f1-per-class vào file pkl
#         metrics_results = {
#             'acc-normal': self.rs_test_acc_normal,
#             'f1-normal': self.rs_test_f1_normal,
#             'f1-per-class-normal': self.rs_test_f1_pc_normal,
#             'train-loss': self.rs_train_loss
#         }
#         with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
#             pickle.dump(metrics_results, f)
#         with open(f'{self.result_path}/models.pkl', 'wb') as f:
#             pickle.dump(self.model_list, f)

import copy
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

from sklearn import metrics
from torch.utils.data import DataLoader, random_split

from utils.data_utils import read_centralized_data


class CentralizedAE(object):

    def __init__(self, args, times):

        # ==========================================================
        # Basic Information
        # ==========================================================
        self.device = args.device
        self.dataset = args.dataset
        self.data_type = args.data_type
        self.global_rounds = args.global_rounds

        self.algo = args.algorithm
        self.exp_name = args.exp_name
        self.seed = args.seed
        self.times = times

        # ==========================================================
        # Model
        # ==========================================================
        self.model = copy.deepcopy(args.model).to(self.device)

        # ==========================================================
        # Data / Training parameters
        # ==========================================================
        self.batch_size = args.batch_size
        self.local_steps = args.local_steps
        self.learning_rate = args.local_learning_rate

        self.input_channels = args.input_channels
        self.image_size = args.image_size
        self.num_classes = args.num_classes

        # ==========================================================
        # Loss & Optimizer
        # ==========================================================
        # self.loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.rec_loss = nn.MSELoss()

        self.lambda_rec = getattr(
            args,
            "lambda_rec",
            1.0
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # ==========================================================
        # READ TRAIN DATA
        #
        # read_centralized_data(..., "train")
        #
        # Expected:
        # full_train_data = union of train data from 20 clients
        # ==========================================================
        full_train_data = read_centralized_data(
            self.dataset,
            data_type="train"
        )

        # ==========================================================
        # Split TRAIN / VALIDATION
        #
        # 90% train
        # 10% validation
        # ==========================================================
        self.train_data, self.val_data = self._split_train_val(
            full_train_data,
            val_ratio=0.1,
            seed=self.seed
        )

        # ==========================================================
        # TRAIN DataLoader
        # ==========================================================
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # ==========================================================
        # VALIDATION DataLoader
        # ==========================================================
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        # ==========================================================
        # READ TEST DATA
        #
        # IMPORTANT:
        # Test data KHÔNG GỘP.
        #
        # test_data_clients[0] -> Client 0 test
        # test_data_clients[1] -> Client 1 test
        # ...
        # test_data_clients[19] -> Client 19 test
        # ==========================================================
        self.test_data_clients = read_centralized_data(
            self.dataset,
            data_type="test"
        )

        # ==========================================================
        # Check number of clients
        # ==========================================================
        self.num_test_clients = len(self.test_data_clients)

        print("\nNumber of test clients:", self.num_test_clients)

        # ==========================================================
        # Create DataLoader for EACH client
        # ==========================================================
        self.test_loaders = []

        for client_id, test_data in enumerate(self.test_data_clients):

            loader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

            self.test_loaders.append(loader)

            print(
                f"Test Client {client_id}: "
                f"{len(test_data)} samples"
            )

        # ==========================================================
        # Early stopping
        # ==========================================================
        self.patience = getattr(
            args,
            "patience",
            10
        )

        self.min_delta = getattr(
            args,
            "min_delta",
            1e-4
        )

        self.early_stop_metric = getattr(
            args,
            "early_stop_metric",
            "f1"
        )

        # ==========================================================
        # History
        # ==========================================================
        self.rs_train_loss = []

        self.rs_test_acc = []
        self.rs_test_precision = []
        self.rs_test_recall = []
        self.rs_test_f1 = []

        # Validation history
        self.rs_val_loss = []
        self.rs_val_f1 = []

        # Per-client test history
        self.client_test_results = []

        # ==========================================================
        # Best model
        # ==========================================================
        self.best_model = None

        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_precision = 0.0
        self.best_recall = 0.0

        self.best_epoch = 0

        # ==========================================================
        # Time
        # ==========================================================
        self.budget = []

        # ==========================================================
        # Save Path
        # ==========================================================
        self.result_path = os.path.join(
            "..",
            "results",
            self.dataset,
            self.data_type,
            self.algo,
            self.exp_name,
            f"seed_{self.seed}"
        )

        os.makedirs(
            self.result_path,
            exist_ok=True
        )

        self.log_file = os.path.join(
            self.result_path,
            "tracking.log"
        )

        # ==========================================================
        # Initial log
        # ==========================================================
        with open(self.log_file, "a") as f:

            f.write("\n")
            f.write("=" * 70)
            f.write("\n")

            f.write(
                f"Algorithm : {self.algo}\n"
            )

            f.write(
                f"Dataset   : {self.dataset}\n"
            )

            f.write(
                f"Seed      : {self.seed}\n"
            )

            f.write(
                f"Device    : {self.device}\n"
            )

            f.write(
                f"Test Clients : {self.num_test_clients}\n"
            )

            f.write("=" * 70)
            f.write("\n")

        # ==========================================================
        # Print dataset information
        # ==========================================================
        print("\nAlgorithm :", self.algo)
        print("Dataset   :", self.dataset)
        print("Train Samples :", len(self.train_data))
        print("Valid Samples :", len(self.val_data))
        print("Test Clients :", self.num_test_clients)

    # ==============================================================
    # Split Train / Validation
    # ==============================================================

    def _split_train_val(
        self,
        dataset,
        val_ratio=0.1,
        seed=0
    ):

        total = len(dataset)

        val_size = max(
            1,
            int(total * val_ratio)
        )

        train_size = total - val_size

        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        return train_set, val_set

    # ==============================================================
    # Process batch
    # ==============================================================

    def _process_batch(self, x, y):

        if isinstance(x, list):
            x = x[0]

        x = x.to(
            self.device,
            non_blocking=True
        )

        y = y.long().to(
            self.device,
            non_blocking=True
        )

        # ==========================================================
        # Reshape image data if necessary
        # ==========================================================
        if (
            self.input_channels > 1
            and len(x.shape) == 2
            and self.image_size > 0
        ):

            x = x.view(
                -1,
                self.input_channels,
                self.image_size,
                self.image_size
            )

        return x, y

    # ==============================================================
    # Forward model
    # ==============================================================

    def _forward(self, x):

        output = self.model(x)

        if isinstance(output, tuple):
            class_output = output[0]
            reconstruction = output[1]

            return class_output, reconstruction

        return output, None

    # ==============================================================
    # Validation
    # ==============================================================

    def evaluate_on_val(self):

        self.model.eval()

        losses = []

        y_true = []
        y_pred = []

        with torch.no_grad():

            for x, y in self.val_loader:

                x, y = self._process_batch(
                    x,
                    y
                )

                output, reconstruction = self._forward(x)

                loss_cls = self.cls_loss(
                    output,
                    y
                )

                loss_rec = self.rec_loss(
                    reconstruction,
                    x
                )

                loss = loss_cls + self.lambda_rec * loss_rec

                losses.append(
                    loss.item()
                )

                pred = torch.argmax(
                    output,
                    dim=1
                )

                y_true.extend(
                    y.detach()
                    .cpu()
                    .numpy()
                )

                y_pred.extend(
                    pred.detach()
                    .cpu()
                    .numpy()
                )

        # ==========================================================
        # Validation metrics
        # ==========================================================

        if len(losses) > 0:
            val_loss = float(
                np.mean(losses)
            )
        else:
            val_loss = 0.0

        val_f1 = metrics.f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        return {
            "loss": val_loss,
            "f1": val_f1
        }

    # ==============================================================
    # Test ONE client
    # ==============================================================

    def evaluate_one_test_client(
        self,
        client_id,
        test_loader
    ):

        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():

            for x, y in test_loader:

                x, y = self._process_batch(
                    x,
                    y
                )

                output, _ = self._forward(x)

                pred = torch.argmax(
                    output,
                    dim=1
                )

                y_true.extend(
                    y.detach()
                    .cpu()
                    .numpy()
                )

                y_pred.extend(
                    pred.detach()
                    .cpu()
                    .numpy()
                )

        y_true = np.asarray(
            y_true
        )

        y_pred = np.asarray(
            y_pred
        )

        # ==========================================================
        # Metrics
        # ==========================================================

        if len(y_true) == 0:

            return {
                "client": client_id,
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "samples": 0
            }

        acc = metrics.accuracy_score(
            y_true,
            y_pred
        )

        precision = metrics.precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = metrics.recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = metrics.f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        return {
            "client": client_id,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "samples": len(y_true)
        }

    # ==============================================================
    # Evaluate TEST on all clients
    #
    # IMPORTANT:
    #
    # 1. Evaluate Client 0 test
    # 2. Evaluate Client 1 test
    # ...
    # 3. Evaluate Client 19 test
    #
    # Then:
    #
    # Weighted Metric =
    #     sum(metric_i * N_i) / sum(N_i)
    #
    # ==============================================================

    def evaluate_on_test_clients(self):

        self.model.eval()

        client_results = []

        total_samples = 0

        weighted_acc = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

        print("\n" + "-" * 70)
        print("Centralized Model - Test on Each Client")
        print("-" * 70)

        with open(
            self.log_file,
            "a"
        ) as f:

            f.write(
                "\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            f.write(
                "Centralized Model - Test on Each Client\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

        # ==========================================================
        # TEST EACH CLIENT
        # ==========================================================

        for client_id, test_loader in enumerate(
            self.test_loaders
        ):

            result = self.evaluate_one_test_client(
                client_id,
                test_loader
            )

            acc = result["acc"]
            precision = result["precision"]
            recall = result["recall"]
            f1 = result["f1"]
            n = result["samples"]

            # ======================================================
            # Weighted sum
            # ======================================================

            weighted_acc += (
                acc * n
            )

            weighted_precision += (
                precision * n
            )

            weighted_recall += (
                recall * n
            )

            weighted_f1 += (
                f1 * n
            )

            total_samples += n

            client_results.append(
                result
            )

            # ======================================================
            # Print
            # ======================================================

            print(
                f"Client {client_id:2d} | "
                f"N={n:8d} | "
                f"Acc={acc:.4f} | "
                f"Precision={precision:.4f} | "
                f"Recall={recall:.4f} | "
                f"F1={f1:.4f}"
            )

            # ======================================================
            # Log
            # ======================================================

            with open(
                self.log_file,
                "a"
            ) as f:

                f.write(
                    f"Client {client_id:2d} | "
                    f"N={n:8d} | "
                    f"Acc={acc:.4f} | "
                    f"Precision={precision:.4f} | "
                    f"Recall={recall:.4f} | "
                    f"F1={f1:.4f}\n"
                )

        # ==========================================================
        # Weighted Average
        # ==========================================================

        if total_samples > 0:

            test_acc = (
                weighted_acc /
                total_samples
            )

            test_precision = (
                weighted_precision /
                total_samples
            )

            test_recall = (
                weighted_recall /
                total_samples
            )

            test_f1 = (
                weighted_f1 /
                total_samples
            )

        else:

            test_acc = 0.0
            test_precision = 0.0
            test_recall = 0.0
            test_f1 = 0.0

        # ==========================================================
        # Save per-client results
        # ==========================================================

        self.client_test_results.append(
            client_results
        )

        # ==========================================================
        # Print weighted result
        # ==========================================================

        print("\nWeighted Average over all clients")

        print(
            f"Accuracy   : {test_acc:.4f}"
        )

        print(
            f"Precision  : {test_precision:.4f}"
        )

        print(
            f"Recall     : {test_recall:.4f}"
        )

        print(
            f"F1-score   : {test_f1:.4f}"
        )

        print(
            f"Total Test Samples : {total_samples}"
        )

        # ==========================================================
        # Log weighted result
        # ==========================================================

        with open(
            self.log_file,
            "a"
        ) as f:

            f.write(
                "\nWeighted Average over all clients\n"
            )

            f.write(
                f"Accuracy   : {test_acc:.4f}\n"
            )

            f.write(
                f"Precision  : {test_precision:.4f}\n"
            )

            f.write(
                f"Recall     : {test_recall:.4f}\n"
            )

            f.write(
                f"F1-score   : {test_f1:.4f}\n"
            )

            f.write(
                f"Total Test Samples : {total_samples}\n"
            )

        return (
            test_acc,
            test_precision,
            test_recall,
            test_f1
        )

    # ==============================================================
    # Train
    # ==============================================================

    def train(self):

        print("\n")
        print("=" * 70)
        print("Start Centralized Training")
        print("=" * 70)

        for round_idx in range(
            self.global_rounds + 1
        ):

            start = time.time()

            print(
                f"\n------------- Round {round_idx} -------------"
            )

            with open(
                self.log_file,
                "a"
            ) as f:

                f.write(
                    f"\n------------- Round {round_idx} -------------\n"
                )

            # ======================================================
            # Train
            # ======================================================

            self.model.train()

            best_state = None
            best_score = -float("inf")
            no_improve = 0

            for epoch in range(
                self.local_steps
            ):

                running_loss = []

                self.model.train()

                for x, y in self.train_loader:

                    x, y = self._process_batch(
                        x,
                        y
                    )

                    output, reconstruction = self._forward(x)

                    # Classification loss
                    loss_cls = self.cls_loss(
                        output,
                        y
                    )

                    # Reconstruction loss
                    loss_rec = self.rec_loss(
                        reconstruction,
                        x
                    )

                    # Total loss
                    loss = loss_cls + self.lambda_rec * loss_rec

                    # Backward
                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()

                    running_loss.append(
                        loss.item()
                    )

                # ==================================================
                # Epoch train loss
                # ==================================================

                if len(running_loss) > 0:

                    epoch_train_loss = float(
                        np.mean(running_loss)
                    )

                else:

                    epoch_train_loss = 0.0

                # ==================================================
                # Validation
                # ==================================================

                val_result = self.evaluate_on_val()

                val_loss = val_result["loss"]
                val_f1 = val_result["f1"]

                # ==================================================
                # Save validation history
                # ==================================================

                self.rs_val_loss.append(
                    val_loss
                )

                self.rs_val_f1.append(
                    val_f1
                )

                # ==================================================
                # Early stopping metric
                # ==================================================

                if self.early_stop_metric == "f1":

                    score = val_f1

                else:

                    score = -val_loss

                # ==================================================
                # Check improvement
                # ==================================================

                if score > (
                    best_score +
                    self.min_delta
                ):

                    best_score = score

                    no_improve = 0

                    best_state = {
                        k: v.detach()
                        .cpu()
                        .clone()
                        for k, v
                        in self.model.state_dict().items()
                    }

                else:

                    no_improve += 1

                # ==================================================
                # Print
                # ==================================================

                print(
                    f"Epoch {epoch + 1}/{self.local_steps} | "
                    f"Train Loss={epoch_train_loss:.4f} | "
                    f"Val Loss={val_loss:.4f} | "
                    f"Val F1={val_f1:.4f}"
                )

                # ==================================================
                # Log
                # ==================================================

                with open(
                    self.log_file,
                    "a"
                ) as f:

                    f.write(
                        f"Epoch {epoch + 1}/{self.local_steps} | "
                        f"Train Loss={epoch_train_loss:.4f} | "
                        f"Val Loss={val_loss:.4f} | "
                        f"Val F1={val_f1:.4f}\n"
                    )

                # ==================================================
                # Early stopping
                # ==================================================

                if no_improve >= self.patience:

                    print(
                        "Early stopping."
                    )

                    with open(
                        self.log_file,
                        "a"
                    ) as f:

                        f.write(
                            "Early stopping.\n"
                        )

                    break

            # ======================================================
            # Restore best model of this round
            # ======================================================

            if best_state is not None:

                self.model.load_state_dict(
                    best_state
                )

                self.model.to(
                    self.device
                )

            # ======================================================
            # TEST
            #
            # Test each of 20 clients
            # then weighted average
            # ======================================================

            (
                test_acc,
                test_precision,
                test_recall,
                test_f1
            ) = self.evaluate_on_test_clients()

            # ======================================================
            # Save test history
            # ======================================================

            self.rs_test_acc.append(
                test_acc
            )

            self.rs_test_precision.append(
                test_precision
            )

            self.rs_test_recall.append(
                test_recall
            )

            self.rs_test_f1.append(
                test_f1
            )

            # ======================================================
            # Save train loss
            # ======================================================

            self.rs_train_loss.append(
                epoch_train_loss
            )

            # ======================================================
            # Time
            # ======================================================

            elapsed = (
                time.time() -
                start
            )

            self.budget.append(
                elapsed
            )

            print(
                f"\nRound {round_idx} Time: "
                f"{elapsed:.2f}s"
            )

            # ======================================================
            # Track best test performance
            # ======================================================

            if test_f1 > self.best_f1:

                self.best_f1 = test_f1

                self.best_acc = test_acc

                self.best_precision = (
                    test_precision
                )

                self.best_recall = (
                    test_recall
                )

                self.best_epoch = (
                    round_idx
                )

                self.best_model = {
                    k: v.detach()
                    .cpu()
                    .clone()
                    for k, v
                    in self.model.state_dict().items()
                }

            # ======================================================
            # GPU memory
            # ======================================================

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        # ==========================================================
        # Restore GLOBAL BEST MODEL
        # ==========================================================

        if self.best_model is not None:

            self.model.load_state_dict(
                self.best_model
            )

            self.model.to(
                self.device
            )

        # ==========================================================
        # Final summary
        # ==========================================================

        print("\n")
        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        print(
            f"Algorithm : {self.algo}"
        )

        print(
            f"Dataset   : {self.dataset}"
        )

        print(
            f"Best Round: {self.best_epoch}"
        )

        print(
            f"Accuracy  : {self.best_acc:.4f}"
        )

        print(
            f"Precision : {self.best_precision:.4f}"
        )

        print(
            f"Recall    : {self.best_recall:.4f}"
        )

        print(
            f"F1-score  : {self.best_f1:.4f}"
        )

        if len(self.budget) > 1:

            avg_time = (
                sum(self.budget[1:]) /
                len(self.budget[1:])
            )

        elif len(self.budget) == 1:

            avg_time = self.budget[0]

        else:

            avg_time = 0.0

        print(
            f"Avg Time/Round : {avg_time:.4f} sec"
        )

        # ==========================================================
        # Save final summary to log
        # ==========================================================

        with open(
            self.log_file,
            "a"
        ) as f:

            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("FINAL RESULT\n")
            f.write("=" * 70 + "\n")

            f.write(
                f"Algorithm : {self.algo}\n"
            )

            f.write(
                f"Dataset   : {self.dataset}\n"
            )

            f.write(
                f"Best Round: {self.best_epoch}\n"
            )

            f.write(
                f"Accuracy  : {self.best_acc:.4f}\n"
            )

            f.write(
                f"Precision : {self.best_precision:.4f}\n"
            )

            f.write(
                f"Recall    : {self.best_recall:.4f}\n"
            )

            f.write(
                f"F1-score  : {self.best_f1:.4f}\n"
            )

            f.write(
                f"Avg Time/Round : "
                f"{avg_time:.4f} sec\n"
            )

        # ==========================================================
        # Save results
        # ==========================================================

        self.save_results()

        # ==========================================================
        # Save best model
        # ==========================================================

        self.save_model()

    # ==============================================================
    # Train metrics
    # ==============================================================

    def train_metrics(self):

        self.model.eval()

        losses = []

        with torch.no_grad():

            for x, y in self.train_loader:

                x, y = self._process_batch(
                    x,
                    y
                )

                output, _ = self._forward(x)

                loss = self.loss(
                    output,
                    y
                )

                losses.append(
                    loss.item()
                )

        return losses

    # ==============================================================
    # Evaluate
    # ==============================================================

    def evaluate(self):

        # ==========================================================
        # Train loss
        # ==========================================================

        train_losses = self.train_metrics()

        if len(train_losses) > 0:

            train_loss = float(
                np.mean(train_losses)
            )

        else:

            train_loss = 0.0

        # ==========================================================
        # Test each client + weighted average
        # ==========================================================

        (
            acc,
            precision,
            recall,
            f1
        ) = self.evaluate_on_test_clients()

        # ==========================================================
        # Save
        # ==========================================================

        self.rs_train_loss.append(
            train_loss
        )

        self.rs_test_acc.append(
            acc
        )

        self.rs_test_precision.append(
            precision
        )

        self.rs_test_recall.append(
            recall
        )

        self.rs_test_f1.append(
            f1
        )

        # ==========================================================
        # Print
        # ==========================================================

        print("\nOverall Performance Metrics")

        print(
            f"\tTrain Loss : {train_loss:.4f}"
        )

        print(
            f"\tAccuracy   : {acc:.4f}"
        )

        print(
            f"\tPrecision  : {precision:.4f}"
        )

        print(
            f"\tRecall     : {recall:.4f}"
        )

        print(
            f"\tF1-score   : {f1:.4f}"
        )

        # ==========================================================
        # Log
        # ==========================================================

        with open(
            self.log_file,
            "a"
        ) as f:

            f.write(
                "\nOverall Performance Metrics\n"
            )

            f.write(
                f"\tTrain Loss : {train_loss:.4f}\n"
            )

            f.write(
                f"\tAccuracy   : {acc:.4f}\n"
            )

            f.write(
                f"\tPrecision  : {precision:.4f}\n"
            )

            f.write(
                f"\tRecall     : {recall:.4f}\n"
            )

            f.write(
                f"\tF1-score   : {f1:.4f}\n"
            )

    # ==============================================================
    # Save results
    # ==============================================================

    def save_results(self):

        metrics_results = {

            "train_loss":
                self.rs_train_loss,

            "val_loss":
                self.rs_val_loss,

            "val_f1":
                self.rs_val_f1,

            "test_acc":
                self.rs_test_acc,

            "test_precision":
                self.rs_test_precision,

            "test_recall":
                self.rs_test_recall,

            "test_f1":
                self.rs_test_f1,

            "client_test_results":
                self.client_test_results,

            "best_acc":
                self.best_acc,

            "best_precision":
                self.best_precision,

            "best_recall":
                self.best_recall,

            "best_f1":
                self.best_f1,

            "best_epoch":
                self.best_epoch,

            "budget":
                self.budget
        }

        save_path = os.path.join(
            self.result_path,
            "metrics_results.pkl"
        )

        with open(
            save_path,
            "wb"
        ) as f:

            pickle.dump(
                metrics_results,
                f
            )

        print(
            "\nMetrics saved to:"
        )

        print(
            save_path
        )

    # ==============================================================
    # Save model
    # ==============================================================

    def save_model(
        self,
        filename="best_model.pt"
    ):

        save_path = os.path.join(
            self.result_path,
            filename
        )

        torch.save(
            self.model.state_dict(),
            save_path
        )

        print(
            f"Model saved to {save_path}"
        )

        with open(
            self.log_file,
            "a"
        ) as f:

            f.write(
                f"\nModel saved to {save_path}\n"
            )

    # ==============================================================
    # Load model
    # ==============================================================

    def load_model(
        self,
        filename="best_model.pt"
    ):

        load_path = os.path.join(
            self.result_path,
            filename
        )

        self.model.load_state_dict(
            torch.load(
                load_path,
                map_location=self.device
            )
        )

        self.model.to(
            self.device
        )

        print(
            f"Loaded model from {load_path}"
        )