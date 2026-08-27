#!/bin/bash
#SBATCH --job-name=training-icfm
#SBATCH --account=rrg-keli
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --mail-user=saurabh_b241170ee@nitc.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=training_CORGI_FULL_CLASS_icfm_sbatch.txt

echo "Launching ICFM 100K Training Job for Corgi Imagenet Class "

# 1. Load system modules safely
module purge
module load StdEnv/2023 python/3.10 scipy-stack/2023b 

# 2. Activate the CORRECT virtual environment
source ~/nibi_env/bin/activate

# 3. Execute the script
python /home/saurabhg/SFU_research_internship/examples/images/LATENT_GENERATION_2/vae_train2.py \
    --model=icfm \
    --dataset_address='/home/saurabhg/scratch/CORGIS_DATASET/img' \
    --output_dir='/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs' \
    --num_color_channels=4 \
    --num_channels=128 \
    --batch_size=100 \
    --epochs=100000 \
    --Save_step=5000 \
    --Dataset_download_flag=False

echo "Training Complete"
