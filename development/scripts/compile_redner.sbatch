#!/bin/bash

#SBATCH -o %x_%j_%N.out   # Output file
#SBATCH -e %x_%j_%N.err	  # Error file
#SBATCH -J RednerCompilation # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1 		  # Run 1 process
#SBATCH --gres=gpu:1	  # Request 1 GPU
#SBATCH --mem=16GB        # 16GiB resident memory pro node

#SBATCH --time=01:00:00   # Expected runtime

#SBATCH --partition=gpu   # Compute on a GPU node

#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.worchel@campus.tu-berlin.de

# Load required modules
module purge
module load cmake/3.16.3
module load comp/gcc/7.2.0
module load nvidia/cuda/10.0

# Enter working directory
echo $PWD
echo "Entering working directory"
cd ~/redner
echo $PWD

# conda create -y --name redner-build-env python=3.6.7 pytorch=1.3 pybind11 scikit-image
source activate redner-build-env

echo "Building redner"
python -u -m pip wheel -w dist --verbose .
exitCode=$?
echo "Finished building (exit code $exitCode)"

conda deactivate

exit $exitCode
