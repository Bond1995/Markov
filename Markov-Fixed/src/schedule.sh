#!/bin/bash
set -e  # exit on error

USER=bondasch
LAB=linx
WANDB_PROJECT="markov-fixed-e0"
WANDB_RUN_GROUP="test01"
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

i=1;
for p in 0.5;
do
    for q in 0.8;
    do
        for e in 0;
        do
            for w in -10 -1 -0.6 0 1 10;
            do
                for b in -10 -1 0 1 10;
                do
                    for j in 1 2 3;
                    do
                        # Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
                        RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;
                        RUN_FILE="python main.py --wandb --wandb_project $WANDB_PROJECT --p $p --q $q --e $e --w $w --b $b"

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

                            if [ `expr $i % 10` -eq 0 ]
                            then
                                sleep 2000;
                            fi
                            i=$((i+1));
                    done
                done
            done
        done
    done
done
        