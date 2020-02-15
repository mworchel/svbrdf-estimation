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
module load nvidia/cuda/10.0

# Switch into correct working directory
echo $PWD
echo "Entering working directory"
cd ~/svbrdf-estimation/development/multiImage_pytorch/
echo $PWD

# conda create -y --name svbrdf-env python=3.6.7 pytorch=1.3 scikit-image matplotlib numpy
# conda run -n svbrdf-env pip install tensorboardX
source activate svbrdf-env

echo "Running training"
# -u option is to use unbuffered stdout and stderr so we can directly tail -f to the outputs
python -u main.py --mode train --input-dir "./../../../materialsData_multi_image/train/" --image-count 0 --model-dir "./models" --epochs 200 --save-frequency 1
exitCode=$?
echo "Finished training (exit code $exitCode)"

conda deactivate

exit $exitCode
