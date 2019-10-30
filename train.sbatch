#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=cst-test
#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=12G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=15:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=claudio.raisch@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

IMG_BASE=/common/singularityImages
IMG_FILE=TCML-Cuda10_0Tensorflow2_0_0_a0.simg
TRAIN_SCRIPT=train_sequence.py
# MODEL=simple_flat__in4_seqlen2_imw256_imh256_out6.h5
MODEL=simple_cnn__in4_seqlen2_imw256_imh256_out6.h5

echo "copying data..."
cp ~/CARLA_Tensorflow/tfrec_sequences/*.zip /scratch/$SLURM_JOB_ID/
cp ~/CARLA_Tensorflow/$TRAIN_SCRIPT /scratch/$SLURM_JOB_ID/$TRAIN_SCRIPT
cp ~/CARLA_Tensorflow/__init__.py /scratch/$SLURM_JOB_ID/__init__.py
cp ~/CARLA_Tensorflow/models/$MODEL /scratch/$SLURM_JOB_ID/$MODEL
mkdir -p /scratch/$SLURM_JOB_ID/src && cp ~/CARLA_Tensorflow/src/config.py /scratch/$SLURM_JOB_ID/src/config.py
cp ~/CARLA_Tensorflow/configs/sample.conf /scratch/$SLURM_JOB_ID/sample.conf
echo "copying simg..."
cp $IMG_BASE/$IMG_FILE /scratch/$SLURM_JOB_ID/$IMG_FILE
echo "copying done, execution following..."

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

cd ~/CARLA_Training/nvidia_out
( watch -n 10 nvidia-smi > nvidia_out.txt ) &
singularity exec --nv /scratch/$SLURM_JOB_ID/$IMG_FILE python3 /scratch/$SLURM_JOB_ID/$TRAIN_SCRIPT /scratch/$SLURM_JOB_ID/sample.conf --base /scratch/$SLURM_JOB_ID
echo DONE!

