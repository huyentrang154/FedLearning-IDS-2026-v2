import argparse
import os
import time
import warnings
import numpy as np
import random
import torch

from flcore.servers.serverAvg import FedAvg
from flcore.servers.serverProx import FedProx
from flcore.servers.serverMOON import MOON
from flcore.servers.serverScaffold import Scaffold
from flcore.servers.serverPer import FedPer
from flcore.servers.serverRep import FedRep
from flcore.servers.serverLG import LG_FedAvg
from flcore.servers.serverGH import FedGH
from flcore.servers.serverALA import FedALA
from flcore.servers.serverAPFL import APFL
from flcore.servers.serverApple import APPLE
from flcore.servers.serverFomo import FedFomo
from flcore.servers.serverUS import FedUS
from flcore.servers.serverHomeLG import FedHomeLG
from flcore.servers.serverHome import FedHome
from flcore.servers.serverMedian import FedMedian
from flcore.servers.serverTrimmedMean import FedTrimmedMean
from flcore.servers.serverFLTrust import FLTrust
from flcore.servers.serverAE import FedAE
from flcore.centralized.centralizedAE import CentralizedAE
from flcore.trainmodel.models import *

warnings.simplefilter("ignore")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    time_list = []
    algorithm = args.algorithm
    for i in range(args.prev, args.times):
        print(f"\n============= Running {i + 1}/{args.times} =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if args.dataset == 'cic_ids':
            n_features, num_classes = 80, 12
        elif args.dataset == 'ctu13_13' or args.dataset == 'ctu13_08':
            n_features, num_classes = 40, 2
        elif args.dataset == 'ctu13_09':
            n_features = 41
        elif args.dataset == 'ctu13_10':
            n_features = 38
        elif args.dataset == 'internet_ad':
            n_features = 1558
        elif args.dataset == 'nslkdd':
            n_features, num_classes = 122, 5
        elif args.dataset == 'spambase':
            n_features = 57
        elif args.dataset == 'unsw':
            n_features, num_classes = 196, 10
        elif args.dataset == 'cifar10':
            # 32x32x3 = 3072
            input_channels, image_size, num_classes = 3, 32, 10
        elif args.dataset == 'bot_iot':
            n_features, num_classes = 34, 5
        elif args.dataset == 'crawl':
            n_features, num_classes = 15, 4
        elif args.dataset == 'toniot':
            n_features, num_classes = 42, 10
        else:
            raise NotImplementedError

        args.num_classes = num_classes
        if args.model_type == 'mlp':
            args.model = MLP(n_features=n_features, num_classes=num_classes).to(args.device)
        elif args.model_type == 'ae':
            args.model = AE(n_features=n_features, num_classes=num_classes).to(args.device)
        elif args.model_type == 'cnn':
            # Bây giờ đã có n_features làm tham số
            args.model = CNN(input_channels=input_channels, image_size=image_size, num_classes=num_classes).to(args.device)
        elif args.model_type == 'cnnae':
            args.model = CNNAE(input_channels=input_channels, image_size=image_size, num_classes=num_classes).to(args.device)

        if algorithm == 'fedavg':
            server = FedAvg(args, i)
        elif algorithm == 'fedae':
            server = FedAE(args, i)
        elif algorithm == 'fedprox':
            server = FedProx(args, i)
        elif algorithm == 'moon':
            server = MOON(args, i)
        elif algorithm == 'scaffold':
            server = Scaffold(args, i)
        elif algorithm == 'fedper':
            server = FedPer(args, i)
        elif algorithm == 'fedrep':
            server = FedRep(args, i)
        elif algorithm == 'lg_fedavg':
            server = LG_FedAvg(args, i)
        elif algorithm == 'fedgh':
            server = FedGH(args, i)
        elif algorithm == 'fedala':
            server = FedALA(args, i)
        elif algorithm == 'apfl':
            server = APFL(args, i)
        elif algorithm == 'apple':
            server = APPLE(args, i)
        elif algorithm == 'fedfomo':
            server = FedFomo(args, i)
        elif algorithm == 'fedus':
            server = FedUS(args, i)
        elif algorithm == 'fedhomelg':
            server = FedHomeLG(args, i)
        elif algorithm == 'fedhome':
            server = FedHome(args, i)
        elif algorithm == 'fedmedian':
            server = FedMedian(args, i)
        elif algorithm == 'fedtrimmedmean':
            server = FedTrimmedMean(args, i)
        elif algorithm == 'fedtrust':
            server = FLTrust(args, i)
        elif algorithm == "centralized":
            server = CentralizedAE(args, i)
        else:
            raise NotImplementedError

        print('Algorithm', algorithm)
        server.train()
        server.save_results()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-algo', "--algorithm", type=str, default='centralized')

    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="nslkdd")
    parser.add_argument('-is', "--image_size", type=int, default=0,
                    help="Kích thước cạnh của ảnh (ví dụ: 32 cho CIFAR-10)")
    parser.add_argument('-ic', "--input_channels", type=int, default=0,
                    help="Số kênh màu của dữ liệu (3 cho RGB, 1 cho Grayscale/Tabular)")
    parser.add_argument('-nb', "--num_classes", type=int, default=12)
    parser.add_argument('-m', "--model_type", type=str, default="ae")
    parser.add_argument('-lbs', "--batch_size", type=int, default=4)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('--round_to_add_clients', type=int, default=0,
                    help="Round number to start including the last 5 clients in training")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
 
    # FedProx
    parser.add_argument('-mu', "--mu", type=float, default=0.1)

    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)

    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)

    parser.add_argument('-dt', "--data_type", type=str, default='noniid')

    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")

    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)

    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)

    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # Label flipping 
    parser.add_argument('-nt', '--noise_type', type=str, default="Asymmetric",
                    choices=[None, 'gaussian_noise', 'label_flipping', 'Asymmetric', 'Symmetric'],
                    help="Type of noise: 'gaussian_noise', 'label_flipping', or None.")
    parser.add_argument('-npw', '--num_of_poisoned_workers', type=float, default=0,
                        help="Number of clients to be poisoned.")
    parser.add_argument('-psr', '--poisoned_sample_ratio', type=float, default=0,
                        help="Ratio of data samples to be poisoned (0 to 1).")
    parser.add_argument('-ans', '--attack_noise_std', type=float, default=0,
                        help="Standard deviation for Gaussian noise.")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    args = parser.parse_args()
    set_random_seed(args.seed)

    print("=" * 60)
    print(f"Random Seed : {args.seed}")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)
# K0VitxF3C#8E

