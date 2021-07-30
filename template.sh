#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={memory}
#SBATCH --mail-type=END
#SBATCH --mail-user=aritraghosh.iem@gmail.com
#SBATCH --partition={gpu}-long
#SBATCH -o /mnt/nfs/scratch1/arighosh/lak/slurm/%j.out
module load python3/current
cd /mnt/nfs/scratch1/arighosh/lak/
source ../venv/simclr/bin/activate
python {train_file}\
    --lr {lr}\
    --setup {setup}\
    --task {task}\
    --fold {fold}\
    --batch_size {batch_size}\
    --dataset {dataset}\
    --model {model}\
    --hidden_dim {hidden_dim}\
    --question_dim {question_dim}\
    --neptune --cuda\
    --name "${SLURM_JOB_ID}" --nodes "${SLURM_JOB_NODELIST}" --slurm_partition "${SLURM_JOB_PARTITION}"
