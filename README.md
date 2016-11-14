# FuseNet
FuseNet is developed as a general architecture for deep convolutional neural network (CNN) to train dataset with RGB-D images. It can be used for semantic segmentation, scene classification and other applications. This repository is an official release of the original work ([see paper](#paper)), and it is implemented based on the [BVLC/caffe](https://github.com/BVLC/caffe) framework.

## Usage
### Installation
The code is compatible with an early Caffe version of June 2016. It is developed under Ubuntu 16.04 with CUDA 7.5 and cuDNN v5.0. If you use the program under other Ubuntu distributions, you may need to comment out line 72--73 in the root CMakeLists.txt file. If you compile under other OS, please use Google as your friend. We mostly test the program on Nvidia Titan X GPU.

### Training and Testing
We provide all needed python scripts and prototxt file to reproduce our published results under `<./fusenet/>`. A short guideline is given below. For detailed information, check [here](fusenet/segmentation/readme.md).

#### Initiazation
Our network architecture is based on VGGNet-16layer. However, since we have extra input channel for depth, we provide the compute the

#### Data preparation
To store dataset, we save paired RGB-D images into LMDB. We also scale the original depth image to the the range of [0, 255]. It is optional further cast the scaled depth value into unsigned char (grayscale), so as to save memory. If you do not want to lose precision, store the scaled depth as float. To prepare LMDB, we provide the following python scripts for your reference. However, you can also write your own image input layer to grab paired RGB-D images.
```
check   ./fusenet/scripts/save_lmdb.py
```

#### LMDB shuffling
We support LMDB shuffling, and recommend to do shuffling after each epoch during training.
To enable this option, flag shuffle to be true for the DataLayer in the prototxt. Note that we do not support shuffling with LevelDB.
```
demo   ./fusenet/segmentation/train_sf1.prototxt
```
#### Weighted cross-entropy loss
On common technique to handle class imbalance is to give the loss of each class a different weight, which typically has a higher value for less frequent classes and a lower value for more frequent class. For semantic segmentation, we support this loss weighting with the SoftmaxWithLossLayer by allowing user to specify a weight for each label. We also provide the script to calculate the weight accordingly to the inverse class frequency.
```
check   ./fusenet/scripts/compute_weight.py
check   ./fusenet/segmentation/train_sf1_weight.prototxt
```

#### Batch normalization
We use batch normalization after each convolution. This is supported by the Caffe BatchNormLayer. Notice that we add ScaleLayer after each BatchNormLayer.


#### Testing
To test the semantic segmentation performance, we provide the python scripts to calculate the global accuracy, average class accuracy and average intersection-over-union score. The implementation is based on confusion matrix.
```
check   ./fusenet/scripts/test_segmentation.py
```



## Released Caffemodel
### Semantic Image Segmentation
#### NYU v2

#### SUN-RGBD

### Scene Classification
To be released.

## <a name="paper">Publication</a>
If you use this code or our trainined model in your work, please consider cite the following paper.

Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016. ([pdf](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf))

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }

## License and Contact
[BVLC/caffe](https://github.com/BVLC/caffe) is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE). The modification to the original code is released under [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).

Contact **Lingni Ma** [:envelope:](lingni@in.tum.de) for questions, comments and reporting bugs.
