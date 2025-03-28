#!/bin/bash

GPU=0

DATA_ROOT='./data/NER_data/i2b2'
DATASET=i2b2
TASK=8-1
INCREMENTAL_METHOD=OURS
STEPS_GLOBAL=5
TASK_NUM=9
EPOCHS_GLOBAL=`expr ${STEPS_GLOBAL} \* ${TASK_NUM}`

EPOCHS_LOCAL=5

SEED=2024
echo ${EPOCHS_GLOBAL}

SCREENNAME="${DATASET}_${TASK}_${INCREMENTAL_METHOD} On GPUs ${GPU}"


RESULTSPATH=results/${DATASET}_${TASK}_${INCREMENTAL_METHOD}/seed_${SEED}


mkdir -p ${RESULTSPATH}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSPATH}"


CUDA_VISIBLE_DEVICES=${GPU} nohup python3 -u fl_main.py --data_root ${DATA_ROOT} --dataset ${DATASET} --task ${TASK} --incremental_method ${INCREMENTAL_METHOD} --epochs_local ${EPOCHS_LOCAL} --steps_global ${STEPS_GLOBAL} --epochs_global ${EPOCHS_GLOBAL} --seed ${SEED} --is_ours --hidd_fea_distill --conloss_prototype --distill_weight 2 --use_entropy_detection >${RESULTSPATH}/res.log 2>&1 &


