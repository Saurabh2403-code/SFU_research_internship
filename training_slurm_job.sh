#!/bin/bash
#Directives to request cluster for the job
#SBATCH --job-name=training-cfm-model

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=32g
#SBATCH --time=20:00
#SBATCH --gpus=1
#SBATCH --mail-user=girisaurabh2020@gmail.com
#SBATCH --output=training_cfm_model_sbatch.txt
echo "Launching  training Job"
module load 
source ~/flow_env/bin/activate
python ~/SFU_research_internhsip/examples/images/cifar10/ImageBaselineExperiment/Cifar10_train_sfu.py --num_channels=32 --batch_size=10 --tiny_dataset_size=50 --epochs=2 --Save_step=1 --Dataset_download_flag=Flase
echo "Done"