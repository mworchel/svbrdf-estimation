# SVBRDF Estimation using a Physically-based Differentiable Renderer

This is the repository to the WS 19/20 computer graphics project "SVBRDF Estimation using a Physically-based Differentiable Renderer" at Technische Universität Berlin (Technical University of Berlin).

In the course of this project, the differentiable path tracer Redner [1] was integrated into the deep network-based SVBRDF estimation pipeline by Deschaintre et al. [2][3].

This repository contains custom PyTorch implementations of the single-view [2] as well as the multi-view method [3]. I used the [reference code](https://github.com/valentin-deschaintre/multi-image-deepNet-SVBRDF-acquisition) as a guidance.

## Getting Started

In order to use the code, you will first need to set up an environment containing the required dependencies. 

If your are using conda, simply run
```
conda env create -f development/multiImage_pytorch/environment.yml
```

If you are using pip, you can install the requirements by running
```
pip install -r development/multiImage_pytorch/requirements.txt 
```

**Warning**: While you will be able to run the code using the official pip package of Redner, the custom patch sampling camera (see documentation) will not be used. In order to enable this feature, you need to manually build and install Redner from source using the [`full-patch-sample-camera` branch](https://github.com/mworchel/redner/tree/full-patch-sample-camera) which is based on Redner 0.3.14.

To run the training procedure on the toy dataset, execute the following scripts in the folder `development/multiImage_pytorch`
```
./train.bat # on Windows
./train.sh  # on Linux-based systems
```

The trained model can by tested by running
```
./test.bat # on Windows
./test.sh  # on Linux-based systems
```

## Implementation Details

The folder `development/multiImage_pytorch` contains the main entry point of the custom implementation. The script `main.py` can be used for training und (very basic) testing of the single-view model. Its usage is roughly outlined in the scripts `test.sh/bat` and `train.sh/bat`. To list available options, run `python main.py --help`.

Here is a short overview of the most important modules:
- `dataset.py`: Contains a class that implements the `torch.utils.data.Dataset` interface and is able to consume the [single-view dataset](https://repo-sam.inria.fr/fungraph/deep-materials/DeepMaterialsData.zip) (~80GB), the [multi-view dataset](https://repo-sam.inria.fr/fungraph/multi_image_materials/supplemental_multi_images/materialsData_multi_image.zip) (~1GB) and folders containing photographs
- `losses.py`: Contains the loss functions used by the pipeline like rendering loss and mixed loss.
- `renderers.py`: Contains a simple differentiable renderer implemented in PyTorch ("in-network") and a renderer that wraps Redner. Both renderers implement the same interface and can be plugged into the rendering loss.
- `environment.py`: Contains classes to set up a scene that can be rendered.
- `models.py`: Contains implementations of the single-view and multi-view networks that follow the `torch.nn.Module` interface.
- `persistance.py`: Contains means to load and save a model for testing or (partitioned) training.

**Note**: The implementation is currently very rough around the edges and contains some legacy code or legacy naming (e.g. "multiImage_pytorch" itself is a misleading name as the code is mainly concerned with the single-view method)

## Additional Material

On top of the code, the repository contains the four presentations that were held during the semester as well as a [website](https://mworchel.github.io/svbrdf-estimation) as project documentation.

## References

[1] Li, T.-M., Aittala, M., Durand, F., Lehtinen, J. 2018. Differentiable Monte Carlo Ray Tracing through Edge Sampling.

[2] Deschaintre, V., Aittala, M., Durand, F., Drettakis, G., Bousseau, A. 2018. Single-Image SVBRDF Capture with a Rendering-Aware Deep Network.

[3] Deschaintre, V., Aittala, M., Durand, F., Drettakis, G., Bousseau, A. 2019. Flexible SVBRDF Capture with a Multi-Image Deep Network. 
