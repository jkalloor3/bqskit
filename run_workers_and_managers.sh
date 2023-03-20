#!/bin/bash
node_id=$(uname -n)
amount_of_gpus=$1
amont_of_workers_per_gpu=$2
total_amount_of_workers=$(($amount_of_gpus * $amont_of_workers_per_gpu))



if [ $amount_of_gpus -eq 0 ]
then
    echo "Will run manager on node $node_id with n args of $amont_of_workers_per_gpu"
    bqskit-manager -n$amont_of_workers_per_gpu -v &> $SCRATCH/bqskit_logs/manager_${SLURM_JOB_ID}_${node_id}.log
    echo "Manager finished on node $node_id"
else
    echo "Starting MPS servers on node $node_id"
    nvidia-cuda-mps-control -d
    for (( gpu_id=0; gpu_id<$amount_of_gpus; gpu_id++ ))
    do
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amont_of_workers_per_gpu &> $SCRATCH/bqskit_logs/workers_${SLURM_JOB_ID}_${node_id}_${gpu_id}.log &
    done

    echo "Will run manager on node $node_id"
    bqskit-manager -x -n$total_amount_of_workers -v &> $SCRATCH/bqskit_logs/manager_${SLURM_JOB_ID}_${node_id}.log
    echo "Manager finished on node $node_id"


    echo "Stop MPS servers on node $node_id"
    echo quit | nvidia-cuda-mps-control
fi
