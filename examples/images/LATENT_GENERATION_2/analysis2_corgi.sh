#!/bin/bash
#SBATCH --job-name=analysis2_of_generated_images
#SBATCH --account=rrg-keli
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=84G
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --mail-user=saurabh_b241170ee@nitc.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=analysis2_of_generated_images.txt

echo "Analysis of generated images "

# 1. Load system modules safely
module purge
module load StdEnv/2023 python/3.10 scipy-stack/2023b 

# 2. Activate the CORRECT virtual environment
source ~/nibi_env/bin/activate
python /home/saurabhg/SFU_research_internship/examples/images/cifar10/metric_distance.py \
    --iteration=2 \
    --generated_images='/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs/icfm/image_tensor/generated_images_13000_iteration5_python_inference_scipt/consolidated_13000.pt'

echo "Training Complete"
