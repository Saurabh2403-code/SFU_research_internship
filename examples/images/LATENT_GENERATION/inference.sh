#!/bin/bash
#SBATCH --job-name=analysis_of_generated_images
#SBATCH --account=rrg-keli
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=84G
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --mail-user=saurabh_b241170ee@nitc.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=analysis_of_generted_images.txt

echo "Analysis of generated images "


module purge
module load StdEnv/2023 python/3.10 scipy-stack/2023b 

source ~/nibi_env/bin/activate

python /home/saurabhg/SFU_research_internship/examples/images/LATENT_GENERATION/inference.py \
    --iteration=2 \
    --output_dir=/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/ \
    --step=95000 
python /home/saurabhg/SFU_research_internship/examples/images/LATENT_GENERATION/inference.py \
    --iteration=3 \
    --output_dir=/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/ \
    --step=95000 
python /home/saurabhg/SFU_research_internship/examples/images/LATENT_GENERATION/inference.py \
    --num_images=3000 \
    --iteration=4 \
    --output_dir=/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/ \
    --step=95000 

echo "Inference Complete"
