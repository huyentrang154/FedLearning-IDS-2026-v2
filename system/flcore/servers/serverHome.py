import copy
import numpy as np
import torch
from flcore.clients.clientHome import *
from utils.data_utils import read_client_data
import os
import pickle
import time


class FedHome(object):
    def __init__(self, args, times):
        self.noise_type = args.noise_type
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.num_clients = args.num_clients
        self.num_classes = args.num_classes
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # for visualize
        self.algo = args.algorithm
        self.rs_train_loss = []
        self.rs_test_acc_normal = []
        self.rs_test_f1_normal = []
        self.rs_test_precision_normal = []
        self.rs_test_recall_normal = []
        
        self.rs_test_acc_flipping = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_flipping = []

        # PHẦN MỚI: Lưu Per-class F1
        self.rs_test_f1_pc_normal = []

        self.times = times
        self.seed = args.seed
        self.exp_name = args.exp_name

        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.poisoned_worker_ids = []
        
        self.initialize_poisoned_workers()
        self.set_clients(args, ClientHome)

        self.global_model = copy.deepcopy(args.model)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        
        self.data_type = args.data_type
        self.train_losses = {i: [] for i in range(self.num_clients)}
        self.result_path = os.path.join('..', 'results', self.dataset, self.data_type, self.algo, self.exp_name, f"seed_{self.seed}")
        os.makedirs(self.result_path, exist_ok=True)

        self.log_file = f"{self.result_path}/tracking.log"
        with open(self.log_file, 'a') as f:
            f.write(f"\nAlgorithm: {self.algo} \nDataset: {self.dataset}\n")
        self.budget = []
        self.model_list = {}

    def initialize_poisoned_workers(self):
        """
        Xác định các client bị tấn công.
        """

        # # Các kiểu nhiễu áp dụng lên toàn bộ client
        # if self.noise_type in ["Asymmetric", "Symmetric"]:
        #     self.poisoned_worker_ids = list(range(self.num_clients))

        # Label flipping / Gaussian chỉ áp dụng theo tỷ lệ
        if self.noise_type in ["label_flipping", "gaussian_noise", "Asymmetric", "Symmetric"] \
                and self.num_of_poisoned_workers > 0:

            num_poisoned = int(self.num_clients * self.num_of_poisoned_workers)

            self.poisoned_worker_ids = np.random.choice(
                range(self.num_clients),
                size=num_poisoned,
                replace=False
            ).tolist()

        else:
            self.poisoned_worker_ids = []

        print(f"Noise Type: {self.noise_type}")
        print(f"Poisoned Worker IDs: {self.poisoned_worker_ids}")

    def train(self):
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            self.evaluate()

            for client in self.selected_clients:
                client.train()

            # Save models
            current_models = [copy.deepcopy(self.global_model)]
            for c in self.clients:
                current_models.append(copy.deepcopy(c.model))
            self.model_list.update({epoch: current_models})

            self.receive_models()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('Round Time:', round(self.budget[-1], 2))

        # --- FINAL SUMMARY ---
        # self.print_final_summary()
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("\nBest performance metrics")
        print("\tacc-flipping:", max(self.rs_test_acc_flipping))
        print("\tacc-normal:", max(self.rs_test_acc_normal))
        # print("\tauc:", max(self.rs_test_auc))
        print("\tf1-flipping:", max(self.rs_test_f1_flipping))
        print("\tf1-normal:", max(self.rs_test_f1_normal))
        print("\trecall-flipping:", max(self.rs_test_recall_flipping))
        print("\trecall-normal:", max(self.rs_test_recall_normal))
        print("\tprecision-flipping:", max(self.rs_test_precision_flipping))
        print("\tprecision-normal:", max(self.rs_test_precision_normal))
        print("\tTime:", sum(self.budget[1:]) / len(self.budget[1:]))

        summary_lines = [
            f"\nAlgorithm: {self.algo} | Dataset: {self.dataset}\n",
        f"\nNoise type: {self.noise_type} | ratio: {self.poisoned_sample_ratio}\n",
            "\nBest performance metrics\n",
            f"\tacc-flipping: {max(self.rs_test_acc_flipping):.4f}\n",
            f"\tacc-normal: {max(self.rs_test_acc_normal):.4f}\n",
            f"\tf1-flipping: {max(self.rs_test_f1_flipping):.4f}\n",
            f"\tf1-normal: {max(self.rs_test_f1_normal):.4f}\n",
            f"\trecall-flipping: {max(self.rs_test_recall_flipping):.4f}\n",
            f"\trecall-normal: {max(self.rs_test_recall_normal):.4f}\n",
            f"\tprecision-flipping: {max(self.rs_test_precision_flipping):.4f}\n",
            f"\tprecision-normal: {max(self.rs_test_precision_normal):.4f}\n",
            f"\tAvg Time/Round: {sum(self.budget[1:]) / len(self.budget[1:]):.4f} sec\n"
        ]

        for line in summary_lines:
            print(line.strip())
            with open(self.log_file, 'a') as f:
                f.write(line)
        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train')
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test')
            client = clientObj(args=args,
                               id=i,
                               poisoned_worker_ids=self.poisoned_worker_ids,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        return list(np.random.choice(self.clients, join_clients, replace=False))

    def send_models(self):
        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        active_train_samples = sum([c.train_samples for c in self.selected_clients])
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / max(1, active_train_samples))
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def aggregate_parameters(self):
        if len(self.uploaded_models) == 0: return
        new_params = copy.deepcopy(self.uploaded_models[0].state_dict())
        for key in new_params.keys():
            new_params[key] = torch.zeros_like(new_params[key])
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                new_params[key] += client_model.state_dict()[key] * w
        self.global_model.load_state_dict(new_params)
    
    def train_metrics(self):
        num_s_n, loss_n, num_s_f, loss_f = [], [], [], []
        for c in self.selected_clients:
            list_loss = c.train_metrics()
            avg_l = sum(list_loss)/len(list_loss)
            self.train_losses[c.id].append(avg_l)
            if c.id in self.poisoned_worker_ids:
                num_s_f.append(len(list_loss)); loss_f.extend(list_loss)
            else:
                num_s_n.append(len(list_loss)); loss_n.extend(list_loss)
        return num_s_n, loss_n, num_s_f, loss_f

    def test_metrics(self):
        #normal
        total_acc_normal = []
        total_precision_normal = []
        total_recall_normal = []
        total_f1_normal = []
        num_samples_normal = 0
        #flipping
        total_acc_flipping = []
        total_precision_flipping = []
        total_recall_flipping = []
        total_f1_flipping = []
        # total_auc = []
        num_samples_flipping = 0

        print('Performance Metrics')
        with open(self.log_file, 'a') as f:
            f.write('Performance Metrics\n')
        for c in self.selected_clients:
            """
            # acc, precision, recall, f1, auc, ns = c.test_metrics()
            # print(
            #     f'\tClient {c.id}: acc {round(acc, 2)}, auc: {round(auc, 2)}, precision: {round(precision, 2)}, '
            #     f'recall: {round(recall, 2)}, f1: {round(f1, 2)}')
            # with open(self.log_file, 'a') as f:
            #     f.write(
            #         f'\tClient {c.id}: acc {round(acc, 2)}, auc: {round(auc, 2)}, precision: {round(precision, 2)}, '
            #         f'recall: {round(recall, 2)}, f1: {round(f1, 2)}\n')
            """
            acc, precision, recall, f1, ns = c.test_metrics()
            print(
                f'\tClient {c.id}: acc {round(acc, 2)}, precision: {round(precision, 2)}, '
                f'recall: {round(recall, 2)}, f1: {round(f1, 2)}')
            with open(self.log_file, 'a') as f:
                f.write(
                    f'\tClient {c.id}: acc {round(acc, 2)}, precision: {round(precision, 2)}, '
                    f'recall: {round(recall, 2)}, f1: {round(f1, 2)}\n')

            if c.id in self.poisoned_worker_ids:
                total_acc_flipping.append(acc * ns)
                # total_auc.append(auc * ns)
                total_recall_flipping.append(recall * ns)
                total_f1_flipping.append(f1 * ns)
                total_precision_flipping.append(precision * ns)
                num_samples_flipping += ns
            else:
                total_acc_normal.append(acc * ns)
                # total_auc.append(auc * ns)
                total_recall_normal.append(recall * ns)
                total_f1_normal.append(f1 * ns)
                total_precision_normal.append(precision * ns)
                num_samples_normal += ns

        return total_acc_normal, total_f1_normal, total_recall_normal, total_precision_normal, num_samples_normal, total_acc_flipping, total_f1_flipping, total_recall_flipping, total_precision_flipping, num_samples_flipping

        # return total_acc, total_auc, total_f1, total_recall, total_precision, num_samples
    def evaluate(self):
        # evaluate train
        stats_train = self.train_metrics()
        train_loss = sum(stats_train[1]) * 1.0 / sum(stats_train[0])  # avg train loss for all clients
        self.rs_train_loss.append(train_loss)

        # evaluate test, avg metrics for all clients
        stats_test = self.test_metrics()
        # test_acc = sum(stats_test[0]) / stats_test[5]
        # test_auc = sum(stats_test[1]) / stats_test[5]
        # test_f1 = sum(stats_test[2]) / stats_test[5]
        # test_recall = sum(stats_test[3]) / stats_test[5]
        # test_precision = sum(stats_test[4]) / stats_test[5]

        if stats_test[4] > 0:
            test_acc_normal = sum(stats_test[0]) / stats_test[4]
            test_f1_normal = sum(stats_test[1]) / stats_test[4]
            test_recall_normal = sum(stats_test[2]) / stats_test[4]
            test_precision_normal = sum(stats_test[3]) / stats_test[4]
        else:
            test_acc_normal = 0
            test_f1_normal = 0
            test_recall_normal = 0
            test_precision_normal= 0

        if stats_test[9] > 0:
            test_acc_flipping = sum(stats_test[5]) / stats_test[9]
            test_f1_flipping = sum(stats_test[6]) / stats_test[9]
            test_recall_flipping = sum(stats_test[7]) / stats_test[9]
            test_precision_flipping = sum(stats_test[8]) / stats_test[9]
        else:
            test_acc_flipping = 0
            test_f1_flipping = 0
            test_recall_flipping = 0
            test_precision_flipping = 0

        self.rs_test_acc_normal.append(test_acc_normal)
        # self.rs_test_auc.append(test_auc)
        self.rs_test_f1_normal.append(test_f1_normal)
        self.rs_test_precision_normal.append(test_precision_normal)
        self.rs_test_recall_normal.append(test_recall_normal)

        self.rs_test_acc_flipping.append(test_acc_flipping)
        # self.rs_test_auc.append(test_auc)
        self.rs_test_f1_flipping.append(test_f1_flipping)
        self.rs_test_precision_flipping.append(test_precision_flipping)
        self.rs_test_recall_flipping.append(test_recall_flipping)

        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("Overall Performance Metrics")
        print("\ttrain loss avg: {:.4f}".format(train_loss))


        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))

    def save_results(self):
        metrics_results = {
            'acc-normal': self.rs_test_acc_normal,
            'f1-normal': self.rs_test_f1_normal,
            'f1-pc-normal': self.rs_test_f1_pc_normal,
            'train-loss': self.rs_train_loss,
            'acc-flipping': self.rs_test_acc_flipping,
            'f1-flipping': self.rs_test_f1_flipping
        }
        with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)
        print(f"Results saved to {self.result_path}")
