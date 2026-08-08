#!/bin/bash
#SBATCH --job-name=training-icfm
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --mail-user=girisaurabh2020@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=training_icfm_sbatch.txt

echo "Launching ICFM 100K Training Job"

# 1. Load system modules
module load StdEnv/2023
module load python/3.10 scipy-stack/2023b 

# 2. Activate virtual environment
source ~/flow_env/bin/activate

# 3. Execute with explicit experimental flags!
python ~/SFU_research_internship/examples/images/cifar10/ImageBaselineExperiment/Cifar10_train_sfu.py \
    --model=icfm \
    --num_channels=128 \
    --batch_size=10 \
    --tiny_dataset_size=100 \
    --epochs=100000 \
    --Save_step=10000 \
    --Dataset_download_flag=False

echo "Training Complete"