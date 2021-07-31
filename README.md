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

This module can be leveraged in any existing point-based segmentation networks. Here we provide an example of how to apply CGA module to CloserLook3D, please refer to [examples/CloserLook3D/README.md](https://github.com/MCG-NJU/CGA-Net/blob/main/examples/CloserLook3D/README.md).

For other backbones, one can try to modify the source code in [CGA/cga.py](https://github.com/MCG-NJU/CGA-Net/blob/main/CGA/cga.py). 


## Acknowledgement

Our Tensorflow code is based on [CloserLook3D](https://github.com/zeliu98/CloserLook3D), [RandLA-Net](https://github.com/QingyongHu/RandLA-Net), [MeteorNet](https://github.com/xingyul/meteornet) and we benefit a lot from [PointNet2](https://github.com/charlesq34/pointnet2), [KPConv](https://github.com/HuguesTHOMAS/KPConv).
