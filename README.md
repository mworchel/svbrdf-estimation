# SVBRDF Estimation using a Physically-based Differentiable Renderer

This is the repository to the WS 19/20 computer graphics project "SVBRDF Estimation using a Physically-based Differentiable Renderer" at Technische Universität Berlin (Technical University of Berlin).

In the course of this project, the differentiable path tracer Redner [1] was integrated into the deep network-based SVBRDF estimation pipeline by Deschaintre et al. [2][3].

## Implementation

Besides the reference implementation of the multi-view method described in [3], this repository contains a custom PyTorch implementation of the multi-view approach as well as an implementation of the single-view method described in [2]. The reference code was merely used as a guidance for the custom implementation and all rights are reserved by the original authors.

The folder `./development/multiImage_pytorch` contains the main entry point of the custom implementation. The script `main.py` can be used for training und (very basic) testing of the single-view model. Its usage is roughly outlined in the scripts `test.sh/bat` and `train.sh/bat`. To list available options, run `python main.py --help`.

Here is a short overview of the most important modules:
- `dataset.py`: Contains a class that implements the `torch.utils.data.Dataset` interface is able to consume the [single-view dataset](https://repo-sam.inria.fr/fungraph/deep-materials/DeepMaterialsData.zip) (~80GB), the [multi-view dataset](https://repo-sam.inria.fr/fungraph/multi_image_materials/supplemental_multi_images/materialsData_multi_image.zip) (~1GB) and folders containing photographs

**Note**: The implementation is currently very rough around the edges and contains some legacy code or legacy naming (e.g. 'multiImage_pytorch' itself is a misleading name as the code is mainly concerned with the single-view method)

## References

[1] Li, T.-M., Aittala, M., Durand, F., Lehtinen, J. 2018. Differentiable Monte Carlo Ray Tracing through Edge Sampling.

[2] Deschaintre, V., Aittala, M., Durand, F., Drettakis, G., Bousseau, A. 2018. Single-Image SVBRDF Capture with a Rendering-Aware Deep Network.

[3] Deschaintre, V., Aittala, M., Durand, F., Drettakis, G., Bousseau, A. 2019. Flexible SVBRDF Capture with a Multi-Image Deep Network. 
