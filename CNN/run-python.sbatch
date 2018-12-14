#!/bin/bash

#SBATCH --job-name=Yelp_CNN
#SBATCH --nodes=1
#SBATCH -c4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xc1113@nyu.edu

module purge
model load python3/intel/3.6.3 cudnn/9.0v7.0.5
model swap cuda cuda/9.2.88
source activate ML

echo
echo "Job starts: $(date)"
echo "Hostname: $(hostname)"

python main.py 

echo
echo "Job ends: $(date)"
echo

rm -rf __pycache__
