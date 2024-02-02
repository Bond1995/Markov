#!/bin/bash
set -e  # exit on error

USER=bondasch
LAB=mlo
WANDB_PROJECT="markov-bce"
WANDB_RUN_GROUP="test01"
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

i=1;
#for p in 0.2 0.4 0.5 0.6 0.8 1.0;
for p in 1.0;
do
    #for q in 0.0 0.2 0.4 0.5 0.6 0.8 1.0;
    for q in 0.8;
    do
        #for sequence_length in 8 16 32 64 128 256 512 1024 2048 4096 8192;
        for sequence_length in 1024;
        #for dropout in 0 0.2;
        do
            for j in 1 2 3 4 5;
            do
                # Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
                RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;
                RUN_FILE="python main.py --wandb --wandb_project $WANDB_PROJECT --p $p --q $q --bce"

                runai submit \
                    --name ${WANDB_RUN_GROUP}-${RUN_ID} \
                    --environment WANDB_PROJECT=$WANDB_PROJECT \
                    --environment WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
                    --environment WANDB_RUN_ID=$RUN_ID \
                    --environment WANDB_API_KEY=$WANDB_API_KEY \
                    --gpu 1 \
                    --image ic-registry.epfl.ch/linx/bondasch-pytorch-base:latest \
                    --large-shm \
                    --host-ipc \
                    --environment DATA_DIR=/home/$USER/data \
                    --environment EPFML_LDAP=$USER \
                    --command -- \
                        /entrypoint.sh \
                        su $USER -c \
                        \"epfml bundle exec $CODE_BUNDLE -- $RUN_FILE\";

                    if [ `expr $i % 6` -eq 0 ]
                        then
                            sleep 2000;
                        fi
                        i=$((i+1));
            done
        done
    done
done
        