# Guideline For Using FuseNet

- [Dataset](#data)
- [Save LMDB](#lmdb)
- [Train FuseNet for semantic segmentation](#train)
- [Test trained FuseNet semantic segmentation](#test)

### <a name="data">Dataset</a>
- NYUv2 RGB-D dataset, download [here](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). This dataset contains 1449 paired RGB-D images. There exists a standard train-val-test split by Gupta et al, see this [paper](https://people.eecs.berkeley.edu/~sgupta/pdf/rcnn-depth.pdf). To map the original labels into 40 classes, we use the mapping by Gupta et al from this [paper](https://people.eecs.berkeley.edu/~sgupta/pdf/GuptaArbelaezMalikCVPR13.pdf). To map the original label into 13 classes, we use the mapping by Couprie et al from this [paper](https://arxiv.org/pdf/1301.3572.pdf). For your convenience, we put the data split and label mapping in the subfolders `./data/`.

   ```
nyuv2_13class_mapping.mat      maps to 13 classes, 0 is invalid
  nyuv2_40class_mapping.mat      maps to 40 classes, 0 is invalid
  nyuv2_splits.mat               standard train-val-test split, index starts from 5001
   ```

- SUN-RGBD dataset, download [here](http://rgbd.cs.princeton.edu/). This dataset consists of 10335 RGB-D images. The standdard trainval-test split uses the first 5050 images for testing and the rest for trainval. We use the default 37 classes provided by the dataset, and again 0 is the invalid.

### <a name="lmdb">Save LMDB</a>

### <a name="train">Train FuseNet for Semantic Segmentation</a>
#### weighted cross-entropy
we calculate the weights using standard trainval subset images by inverse class frequency. You can find these weights in the following files and optionally use them during training.
```
check    ./data/nyuv2_13class_weight.txt
check    ./data/nyuv2_40class_weight.txt
check    ./data/sunrgb_37class_weight.txt
```

### <a name="test">Test Semantic Segmentation</a>
