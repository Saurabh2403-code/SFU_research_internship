#!/bin/bash
#SBATCH --job-name=analysis_of_generated_images
#SBATCH --account=rrg-keli
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=84G
#SBATCH --time=10:00:00
#SBATCH --gpus=h100:1
#SBATCH --mail-user=saurabh_b241170ee@nitc.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=analysis_of_generted_images.txt

echo "Analysis of generated images "


module purge
module load StdEnv/2023 python/3.10 scipy-stack/2023b 

source ~/nibi_env/bin/activate

python /home/saurabhg/SFU_research_internship/examples/images/cifar10/metric_distance.py \
    --iteration=1 \
    --generated_images='/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/image_tensor/generated_images_1000_iteration1_python_inference_scipt.pt'

echo "Iteration1 Analysis Complete"

python /home/saurabhg/SFU_research_internship/examples/images/cifar10/metric_distance.py \
    --iteration=2 \
    --generated_images='/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/image_tensor/generated_images_1000_iteration2_python_inference_scipt.pt'
echo "Iteration2 Analysi Complete"

python /home/saurabhg/SFU_research_internship/examples/images/cifar10/metric_distance.py \
    --iteration=3 \
    --generated_images='/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/image_tensor/generated_images_1000_iteration3_python_inference_scipt.pt'

python /home/saurabhg/SFU_research_internship/examples/images/cifar10/metric_distance.py \
    --iteration=4 \
    --generated_images='/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/image_tensor/generated_images_3000_iteration4_python_inference_scipt.pt'
echo "Iteration3 Analysis Complete"

echo "Analysis Complete"
