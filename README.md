# CGA-Net: Category Guided Aggregation for Point Cloud Semantic Segmentation
by [Lu Tao](https://github.com/inspirelt), [Wang Limin](https://wanglimin.github.io/)

```
@inproceedings{lu2021cga,
  title={CGA-Net: Category Guided Aggregation for Point Cloud Semantic Segmentation},
  author={Lu, Tao and Wang, Limin and Wu, Gangshan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11693--11702},
  year={2021}
}
```

## Introduction

This is the official implementation of [CGA-Net: Category Guided Aggregation for Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_CGA-Net_Category_Guided_Aggregation_for_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf), which proposes to utilize different aggregation strategies between the same category and different categories. Here we provide the Tensorflow version. The code is very clear and easy to transplant to other frameworks.

## Usage

This module can be leveraged in any existing point-based segmentation networks. Here we provide an example of how to apply CGA module to CloserLook3D, which can be run as follows:

`cd examples/CloserLook3D`
`sh train_s3dis.sh`

For other backbones, one can try to modify the source code in [CGA/cga.py](https://github.com/MCG-NJU/CGA-Net/blob/main/CGA/cga.py). 

## Main Results


## Acknowledgement

Our Tensorflow code is based on [CloserLook3D](), [RandLA-Net](), [MeteorNet]() and we benefit a lot from [PointNet++](), [KPConv]().
