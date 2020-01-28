#!/bin/bash

#SBATCH -o %x_%j_%N.out   # Output file
#SBATCH -e %x_%j_%N.err	  # Error file
#SBATCH -J SVBRDFTraining # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1 		  # Run 1 process
#SBATCH --gres=gpu:1	  # Request 1 GPU
#SBATCH --mem=16GB        # 16GiB resident memory pro node

#SBATCH --time=48:00:00   # Expected runtime

#SBATCH --partition=gpu   # Compute on a GPU node

#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.worchel@campus.tu-berlin.de

# Load required modules
module purge

module load comp/gcc/7.2.0
module load python/3.6.7

echo $PWD
echo "Entering working directory"

cd ~/svbrdf-estimation/development/multiImage_pytorch/
echo $PWD

echo "Run training"
python3 -u main.py --mode train --input-dir "./../../../materialsData_multi_image/train/" --image-count 0 --model-dir "./models" --epochs 200 --save-frequency 1
exitCode=$?
echo "Finished training (exit code $exitCode)"

exit $exitCode
