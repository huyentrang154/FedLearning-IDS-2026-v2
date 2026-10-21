#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=6

DEVICE=cuda
DEVICE_ID=0

EXP_NAME=$(date +"%Y%m%d-%H%M%S")
NOISYTYPE = Asymmetric

mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    nslkdd
    unsw
    cic_ids
)

#########################################
# Algorithms
#########################################

ALGOS=(
    fedavg
    fedgh
    fedrep
    fedhome
    lg_fedavg
    fedus
)

#########################################
# Seeds
#########################################

SEEDS= 0


#########################################
# Fixed parameters
#########################################

POISON_WORKERS= 1.0

RATIO = (
    0.1
    0.2
    0.4 
)
#########################################

run_job(){

    DATASET=$1
    ALGO=$2
    RATIO=$3

    echo "================================================="
    echo "Dataset : $DATASET"
    echo "Algorithm : $ALGO"
    echo "Poisoned sample ratio : $RATIO"
    echo "================================================="

    python3 main.py \
        --algorithm $ALGO \
        --dataset $DATASET \
        --device $DEVICE \
        --device_id $DEVICE_ID \
        --seed $SEED \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $RATIO \
        --noise_type $NOISYTYPE \
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED}"

}

#########################################

for DATASET in "${DATASETS[@]}"
do
    for ALGO in "${ALGOS[@]}"
    do
        for RATIO in "${RATIO[@]}"
        do

            while [ $(jobs -pr | wc -l) -ge $MAX_JOBS ]
            do
                sleep 3
            done

            run_job "$DATASET" "$ALGO" "$RATIO" &

        done
    done
done

wait

echo
echo "==========================================="
echo "All experiments finished."
echo "==========================================="