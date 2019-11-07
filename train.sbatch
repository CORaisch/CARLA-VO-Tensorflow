#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=cst
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

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

IMG_FILE=/common/singularityImages/TCML-Cuda10_0Tensorflow2_0_0.simg
HOME=/home/raisch/CARLA_Tensorflow

cd ~/CARLA_Training/nvidia_out
( watch -n 10 nvidia-smi > nvidia_out.txt ) &
singularity exec --nv $IMG_FILE python3 $HOME/train_sequence.py $HOME/configs/cluster.conf --unpack_to /scratch/$SLURM_JOB_ID
echo DONE!

