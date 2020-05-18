# SVBRDF Estimation using a Physically-based Differentiable Renderer

This is the repository to the WS 19/20 computer graphics project "SVBRDF Estimation using a Physically-based Differentiable Renderer" at Technische Universität Berlin (Technical University of Berlin).

In the course of this project, the differentiable path tracer Redner [1] was integrated into the deep network-based SVBRDF estimation pipeline by Deschaintre et al. [2][3].
## Implementation

This repository contains custom PyTorch implementations of the single-view [2] as well as the multi-view method [3]. The [reference code](https://repo-sam.inria.fr/fungraph/multi_image_materials/supplemental_multi_images/multiImage_code.zip) was merely used as a guidance for the custom implementation and all rights are reserved by the original authors.

The folder `./development/multiImage_pytorch` contains the main entry point of the custom implementation. The script `main.py` can be used for training und (very basic) testing of the single-view model. Its usage is roughly outlined in the scripts `test.sh/bat` and `train.sh/bat`. To list available options, run `python main.py --help`.

Here is a short overview of the most important modules:
- `dataset.py`: Contains a class that implements the `torch.utils.data.Dataset` interface is able to consume the [single-view dataset](https://repo-sam.inria.fr/fungraph/deep-materials/DeepMaterialsData.zip) (~80GB), the [multi-view dataset](https://repo-sam.inria.fr/fungraph/multi_image_materials/supplemental_multi_images/materialsData_multi_image.zip) (~1GB) and folders containing photographs
- `losses.py`: Contains the loss functions used by the pipeline like rendering loss and mixed loss.
- `renderers.py`: Contains a simple differentiable renderer implemented in PyTorch ("in-network") and a renderer that wraps Redner. Both renderers implement the same interface and can be plugged into the rendering loss.
- `environment.py`: Contains classes to set up a scene that can be rendered.
- `models.py`: Contains implementations of the single-view and multi-view networks that follow the `torch.nn.Module` interface.
- `persistance.py`: Contains means to load and save a model for testing or (partitioned) training.

**Note**: The implementation is currently very rough around the edges and contains some legacy code or legacy naming (e.g. "multiImage_pytorch" itself is a misleading name as the code is mainly concerned with the single-view method)

## Additional Material

On top of the code, the repository contains the four presentations that were held during the semester as well as a website that serves as project documentation. The website can be accessed under the following link:
[https://mworchel.github.io/projects/cgp/index.html](https://mworchel.github.io/projects/cgp/index.html)

## Requirements

The following Python packages are required to run the code (tested version in parentheses)

- cv2 (4.1.0.25)
- json (0.8.5)
- Pillow (6.1.0)
- redner (0.3.14) 
    - The official pip package does **not** work since we require a new camera type. The `full-patch-sample-camera` branch in [this fork](https://github.com/mworchel/redner/tree/full-patch-sample-camera) must be used
- torch (1.3.0)
- numpy (1.16.4)
- tensorboardX (1.9)

## References

[1] Li, T.-M., Aittala, M., Durand, F., Lehtinen, J. 2018. Differentiable Monte Carlo Ray Tracing through Edge Sampling.

[2] Deschaintre, V., Aittala, M., Durand, F., Drettakis, G., Bousseau, A. 2018. Single-Image SVBRDF Capture with a Rendering-Aware Deep Network.

[3] Deschaintre, V., Aittala, M., Durand, F., Drettakis, G., Bousseau, A. 2019. Flexible SVBRDF Capture with a Multi-Image Deep Network. 
