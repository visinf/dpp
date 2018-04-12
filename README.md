
# Detail Preserving Pooling in Torch 

This repository contains the code for DPP introduced in the following paper:<br>
**[Detail-Preserving Pooling in Deep Networks](https://arxiv.org/abs/1804.04076) (CVPR 2018)**<br>
[Faraz Saeedan](http://www.visinf.tu-darmstadt.de/team_members/fsaeedan/fsaeedan.en.jsp)<sup>1</sup>, [Nicolas Weber](https://www.mergian.de)<sup>1,2*</sup>, [Michael Goesele](https://www.gcc.tu-darmstadt.de/home/we/michael_goesele/)<sup>1,3*</sup>, and [Stefan Roth](http://www.visinf.tu-darmstadt.de/team_members/sroth/sroth.en.jsp)<sup>1</sup><br>
<sup>1</sup>TU Darmstadt ([VISINF](http://www.visinf.tu-darmstadt.de/visinf/news/index.en.jsp) & [GCC](https://www.gcc.tu-darmstadt.de/home/index.en.jsp)),  <sup>2</sup>NEC Laboratories Europe,  <sup>3</sup>Oculus Research<br>
<sup>*</sup>Work carried out while at TU Darmstadt </p>


## Citation
If you find DPP useful in your research, please cite:

	@inproceedings{saeedan2018dpp,
	  title={Detail-preserving pooling in deep networks},
	  author={Saeedan, Faraz and Weber, Nicolas and Goesele, Michael and Roth, Stefan},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2018}
	}

<center> <img src="./flowchart.png" width="800"> </center>


## Requirements
- This code is built on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch). Check that repository for requirements and preparations.
- In addition you need the [nnlr](https://github.com/gpleiss/nnlr) package installed in Torch.

## Install
 1. Install the visinf package first. This package will give you access to all the models that were used in the paper: Symmetric, asymmetric, full, lite, with and without stochasticity. Please check the reference paper for a description of these variants.
 2. Clone the repository:

    ```bash
    git clone https://github.com/visinf/dpp.git 
    ```
    
 3. Follow one of the recipes below or try your own.

## Usage
For training, simply run main.lua. By default, the script runs ResNet-110 with $DPP sym lite$ on CIFAR10 with 1 GPU and 2 data-loader threads.
To run the models used in **CIFAR10** experiments try:

```bash
    th main.lua -data [imagenet-folder with train and val folders] -netType resnetdpp -poolingType DPP_sym_lite -save [folder to save results] -stochasticity 'false' -manualSeed xyz
```
```bash
    th main.lua -data [imagenet-folder with train and val folders] -netType resnetdpp -poolingType DPP_asym_lite -save [folder to save results] -stochasticity 'false' -manualSeed xyz
```
```bash
    th main.lua -data [imagenet-folder with train and val folders] -netType resnetdpp -poolingType DPP_sym_full -save [folder to save results] -stochasticity 'false' -manualSeed xyz
```
```bash
    th main.lua -data [imagenet-folder with train and val folders] -netType resnetdpp -poolingType DPP_asym_full -save [folder to save results] -stochasticity 'false' -manualSeed xyz
```
```bash
    th main.lua -data [imagenet-folder with train and val folders] -netType resnetdpp -poolingType DPP_sym_lite -save [folder to save results] -stochasticity 'true' -manualSeed xyz
```
replace *xyz* with your desired random number generator seed.

To train ResNets on **ImageNet** try:
```bash
    th main.lua -depth 50 -batchSize 85 -nGPU 4 -nThreads 8 -shareGradInput true -data [imagenet-folder] -dataset imagenet -LR 0.033 -netType resnetdpp -poolingType DPP_sym_lite
```
or
```bash
    th main.lua -depth 101 -batchSize 85 -nGPU 4 -nThreads 8 -shareGradInput true -data [imagenet-folder] -dataset imagenet -LR 0.033 -netType resnetdpp -poolingType DPP_sym_lite
```

## Different implementations of DPP
There are two different implementation codes of DPP available in the Visinf package:

1. Black-box implementation: For this implementation we have derived closed form equations for the forward and backward passes of the inverse bilateral pooling component of DPP and implemented them in CUDA. **visinf.SpatialInverseBilateralPooling** gives access to this implementation, which is very fast and memory efficient. This version can be used for large-scale experiments such as ImageNet, but gives very little insight into various elements inside the block and modifying them is difficult and requires re-deriving and implementing the gradients w.r.t. parameters and inputs.

2. Implementation based on [nngraph](https://github.com/torch/nngraph): This version is made up of a number of Torch primitive blocks connected to each other in a graph. The internals of the block can be understood easily, examined, and altered if need be. This version is fast but not memory efficient for large image sizes. The memory overhead for CIFAR-sized experiments is moderate.

## Contact
If you have further questions or discussions write an email to faraz.saeedan@visinf.tu-darmstadt.de
