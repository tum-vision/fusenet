# FuseNet
## [[PyTorch]](https://github.com/zanilzanzan/FuseNet_PyTorch)
Please refer to PyTorch implementation for an up-to-date implementation.

FuseNet is developed as a general architecture for deep convolutional neural network (CNN) to train dataset with RGB-D images. It can be used for semantic segmentation, scene classification and other applications. This repository is an official release of [this paper](#paper), and it is implemented based on the [BVLC/caffe](https://github.com/BVLC/caffe) framework.

* [Using FuseNet](#usage)
* [Released Caffemodel](#release)
* [Publication](#paper)
* [License and Contact](#other)

## Usage
### Installation
The code is compatible with an early Caffe version of June 2016. It is developed under Ubuntu 16.04 with CUDA 7.5 and cuDNN v5.0. If you use the program under other Ubuntu distributions, you may need to comment out line 72--73 in the root CMakeLists.txt file. If you compile under other OS, please use Google as your friend. We mostly test the program with Nvidia Titan X GPU. Please note multi-GPU training is supported.
```
git clone https://github.com/tum-vision/fusenet.git
mkdir build && cd build
cmake ..
make -j10
make runtest -j10
```

### Training and Testing
We provide all needed python scripts and prototxt files to reproduce our published results under `./fusenet/`. A short guideline is given below. For further detailed instructions, check [here](fusenet/readme.md).

#### Initiazation
Our network architecture is based on VGGNet-16layer. However, since we have extra input channel for depth, we provide the compute the

#### Data preparation
To store dataset, we save paired RGB-D images into LMDB. We also scale the original depth image to the the range of [0, 255]. It is optional further cast the scaled depth value into unsigned char (grayscale), so as to save memory. If you do not want to lose precision, store the scaled depth as float. To prepare LMDB, we provide the following python scripts for your reference. However, you can also write your own image input layer to grab paired RGB-D images.
```
demo   ./fusenet/scripts/save_lmdb.py
```

#### LMDB shuffling
We support LMDB shuffling, and recommend to do shuffling after each epoch during training.
To enable this option, flag shuffle to be true for the DataLayer in the prototxt. Note that we do not support shuffling with LevelDB.
```
demo   ./fusenet/segmentation/nyuv2_sf1/train.prototxt
```
#### Weighted cross-entropy loss
On common technique to handle class imbalance is to give the loss of each class a different weight, which typically has a higher value for less frequent classes and a lower value for more frequent class. For semantic segmentation, we support this loss weighting with the SoftmaxWithLossLayer by allowing user to specify a weight for each label. One way to set the weights is accordingly to the inverse class frequency (see our paper for detail). We provide the weights used in our paper in `./fusenet/data/`.

#### Batch normalization
We use batch normalization after each convolution. This is supported by the Caffe BatchNormLayer. Notice that we add ScaleLayer after each BatchNormLayer.


#### Testing
To test the semantic segmentation performance, we provide the python scripts to calculate the global accuracy, average class accuracy and average intersection-over-union score. The implementation is based on confusion matrix.
```
demo   ./fusenet/scripts/test_segmentation.py
```

## <b name="release">Released Caffemodel</b>
### Semantic Image Segmentation
The items marked with ticks are already available for downloading, otherwise they will be released soon. Unless otherwise stated, all models are finetuned from pretrained VGGNet-16Layer model. Stay tuned :fire:

#### NYUv2 40-class semantic segmentation
More information about the dataset, check [here](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
* FuseNet-SF1:

     This model is trained with the FuseNet Sparse-Fusion1 architecture on 320x240 resolution. To obtain 640x480 full resolution, you can use bilinear upsample the segmenation or better with CRF refinement.
 - [x] [deploy.prototxt](fusenet/segmentation/nyu40-sf1/deploy.prototxt)
 - [ ] caffemodel

* FuseNet-SF5:

    This model is trained the FuseNet SparseFusion5 architecture on 320x240 resolution. It gives 66.0% global pixelwise accuracy, 43.4% average classwise accuracy and 32.7% average classwise IoU.
 - [x] [deploy.prototxt](fusenet/segmentation/nyu40-sf5/deploy.prototxt)
 - [ ] caffemodel

#### SUN-RGBD 37-class semantic segmentation
More information about the dataset, check [here](http://rgbd.cs.princeton.edu/).
* FuseNet-SF5:

    This model is trained with 224x224 resolution. It gives 76.3% global pixelwise accuracy, 48.30% average classwise accuracy and 37.3% average classwise IoU,
 - [x] [deploy.prototxt](fusenet/segmentation/sunrgbd-sf5/deploy.prototxt)
 - [ ] caffemodel


### Scene Classification
To be released.

## <c name="paper">Publication</c>
If you use this code or our trainined model in your work, please consider cite the following paper.

Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016. ([pdf](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf))

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }

## <d name="others"> License and Contact</d>
[BVLC/caffe](https://github.com/BVLC/caffe) is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE). The modification to the original code is released under [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).

Contact **Lingni Ma** [:envelope:](mailto:lingni@in.tum.de) for questions, comments and reporting bugs.
