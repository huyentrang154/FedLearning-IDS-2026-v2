# Introduction

This is the implementation of our method


# Dataset

Here, we upload IoT datasets: ctu13_08, cic_ids
To generate data, run command line:

```
cd dataset
# python generate_cic_ids.py iid - - # for iid and unbalanced scenario
# python generate_cic_ids.py iid balance - # for iid and balanced scenario
# python generate_cic_ids.py noniid - pat # for pathological noniid and unbalanced scenario
# python generate_cic_ids.py noniid - dir # for practical noniid and unbalanced scenario
python generate_cic_ids.py balance
python generate_ctu13_13.py balance
python generate_nslkdd.py balance
python generate_internet_ad.py balance
python generate_spambase.py balance
python generate_unsw.py balance
python generate_ctu13_10.py balance
python generate_ctu13_09.py balance
python generate_unsw_big.py balance
```

# System

- `main.py`: configurations of methods. 
- `run_me.sh`: start methods. 
- `env_linux.yaml`: python environment to run method on Linux. 
- `./flcore`: 
    - `./clients/client.py`: the code on the client. 
    - `./servers/server.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for backbones/ models. 
- `./utils`:
    - `data_utils.py`: the code to read the dataset.

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *fl*. 
```
conda env create -f env_cuda_latest.yaml # for Linux
```


## Training and Evaluation

All codes corresponding to methods are stored in `./system`. Just run the following commands.

```
cd ./system
sh run_me.sh
# python main.py -data cic_ids -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo fedavg
python main.py -data cic_ids -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo fedprox -mu 0.1
python main.py -data ctu13_13 -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo moon -mu 0.1 -tau 0.1
python main.py -data nslkdd -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo fedala -dt noniid
python main.py -data nslkdd -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo apfl -dt noniid
python main.py -data ctu13_13 -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo fedus -dt noniid
python .\main.py -data crawl -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo lg_fedavg -dt noniid -m mlp
python main.py -data nslkdd -lbs 8 -lr 0.005 -gr 50 -ls 5 -algo fedhomelg -dt noniid -m ae

```
