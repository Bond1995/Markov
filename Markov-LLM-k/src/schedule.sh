#!/bin/bash
set -e  # exit on error

USER=bondasch
LAB=linx
WANDB_PROJECT="markov-LLM-layers-iclr"
WANDB_RUN_GROUP="test01"
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

i=1;
for chain in switch;
do
    for order in 1;
    do
        for n_layer in 1 2 4 8;
        do
            for n_head in 1;
            do
                #for n_embd in 16 32 64;
                for n_embd in 8;
                do
                    #for batch_size in 16 50 100;
                    for batch_size in 16;
                    do
                        #for sequence_length in 8 16 32;
                        for sequence_length in 512;
                        do
                            for iterations in 500;
                            do
                                for j in 1 2 3;
                                do
                                    # Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
                                    RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;
                                    RUN_FILE="python main.py --wandb --wandb_project $WANDB_PROJECT --chain $chain --p 0.5 --q 0.8 --order $order --batch_size $batch_size --n_embd $n_embd --n_layer $n_layer --n_head $n_head --sequence_length $sequence_length --iterations $iterations"

                                    runai-rcp submit \
                                        --name ${WANDB_RUN_GROUP}-${RUN_ID} \
                                        --environment WANDB_PROJECT=$WANDB_PROJECT \
                                        --environment WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
                                        --environment WANDB_RUN_ID=$RUN_ID \
                                        --environment WANDB_API_KEY=$WANDB_API_KEY \
                                        --gpu 1 \
                                        --image ic-registry.epfl.ch/linx/bondasch-pytorch-base:latest \
                                        --environment DATA_DIR=/home/$USER/data \
                                        --environment EPFML_LDAP=$USER \
                                        --command -- \
                                            /entrypoint.sh \
                                            su $USER -c \
                                            \"epfml bundle exec $CODE_BUNDLE -- $RUN_FILE\";

                                    if [ `expr $i % 13` -eq 0 ]
                                    then
                                        sleep 5400;
                                    fi
                                    i=$((i+1));
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
