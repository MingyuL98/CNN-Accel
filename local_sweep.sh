#!/bin/bash

# replace with your python script
EXE="bash task.sh"
NUM_GPUS=4
LOG_DIR="./logs"

#echo "Enter a run name: (no spaces or slashes)"
#read RUN_NAME

# we create a new short name for each experiment run
EXPERIMENT_HASH=$(date +%s%N | sha256sum | head -c 6)
EXP_NAME="Sweeping_Task1"_${EXPERIMENT_HASH}
OUTPUT_DIR="${LOG_DIR}/${EXP_NAME}"
mkdir -p $OUTPUT_DIR

echo "Running new experiments: $EXP_NAME"

# MODIFY THESE LINES WITH THE ARGUMENTS YOU NEED
# create your argument combinations here
arg_queue=()

for type in "channel" "filter"; do 
    for ln_norm in "l1" "l2"; do
        for amount in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8; do
            arg_queue+=(" ${type} ${amount} ${ln_norm}")
        done
    done
done


# run the queue here
job_counter=0
device_counter=0
for args in "${arg_queue[@]}"; do

    # set the correct cuda device
    # export CUDA_VISIBLE_DEVICES="$device_counter"

    # log the job start
    job_start=$(date +"%m-%d %H:%M:%S")
    echo $job_counter $job_start $EXE $args | sed "s/ /, /g" >> ${OUTPUT_DIR}/job_list.csv
    echo "Launched $job_counter @ $job_start on $device_counter"

    # launch the job
    $($EXE $args > ${OUTPUT_DIR}/${job_counter}.log) &

    # increment the counters
    ((device_counter++))
    ((job_counter++))

    # wait for the current batch of jobs to complete
    # not super efficient, but simple
    if [[ ${device_counter} -eq ${NUM_GPUS} ]]; then
        wait
        device_counter=0
    fi
done