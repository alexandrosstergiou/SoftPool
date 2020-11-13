# Refining activation downsampling with SoftPool
![supported versions](https://img.shields.io/badge/python-3.5%2C3.6-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue?logo=Pytorch)
<!---![Library](https://img.shields.io/badge/library-TensorFlow-orange?logo=Tensorflow)--->
<!---![Library](https://img.shields.io/badge/library-Keras-red?logo=Keras)--->

![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)


--------------------------------------------------------------------------------
## Abstract
 In every Convolutional Neural Network (CNN), pooling is used to decrease the size of activation maps. This process is crucial to locally achieve spatial invariance and to increase the receptive field of subsequent convolutions. Pooling operations should minimize the loss of information in the activation maps. At the same time, the computation and memory overhead should be limited. To meet these requirements, we propose SoftPool: a fast and efficient method that sums exponentially weighted pixels. Compared to a range of other pooling methods, SoftPool retains more of the information in the downsampled activation maps. More refined downsampling leads to better classification accuracy. On ImageNet1K, for a range of popular CNN architectures, replacing the original pooling operations by SoftPool leads to consistent improvements in the order of 1-2\%. We also demonstrate the merits of SoftPool on video datasets for action recognition. Again, replacing only the pooling layers consistently increases accuracy while computational load and memory remain limited. These favorable properties make SoftPool an excellent replacement for current pooling operations, including max-pool and average-pool. <p align="center">

<i></i>
<br>
<p align="center">
<a href="#" target="blank" >[arXiv preprint - Link coming soon...]</a>
</p>

Image based pooling. Images are sub-sampled in both height and width by half.

|Original|<img src="images/buildings.jpg" width="130" />|<img src="images/otters.jpg" width="130" />|<img src="images/tennis_ball.jpg" width="130" />|<img src="images/puffin.jpg" width="130" />|<img src="images/tram.jpg" width="130" />|<img src="images/tower.jpg" width="130" />|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Soft Pool|<img src="images/buildings_soft.jpg" width="130" />|<img src="images/otters_soft.jpg" width="130" />|<img src="images/tennis_ball_soft.jpg" width="130" />|<img src="images/puffin_soft.jpg" width="130" />|<img src="images/tram_soft.jpg" width="130" />|<img src="images/tower_soft.jpg" width="130" />|

Video based pooling. Videos are sub-sampled in time, height and width by half.


|Original|<img src="images/cars.gif" width="130" />|<img src="images/basketball.gif" width="130" />|<img src="images/parkour.gif" width="130" />|<img src="images/bowling.gif" width="130" />|<img src="images/pizza_toss.gif" width="130" />|<img src="images/pass.gif" width="130" />|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Soft Pool|<img src="images/cars_soft.gif" width="130" />|<img src="images/basketball_soft.gif" width="130" />|<img src="images/parkour_soft.gif" width="130" />|<img src="images/bowling_soft.gif" width="130" />|<img src="images/pizza_toss_soft.gif" width="130" />|<img src="images/pass_soft.gif" width="130" />|

## Dependencies
All parts of the code assume that `torch` is of version 1.4 or higher. There might be instability issues on previous versions.

> ***! Disclaimer:*** This repository is heavily structurally influenced on Ziteng Gao's LIP repo [https://github.com/sebgao/LIP](https://github.com/sebgao/LIP)

## Installation

You can build the repo through the following commands:
```
$ cd SoftPool-master/pytorch
$ make install
--- (optional) ---
$ make test
```


## Usage

You can load any of the 1D, 2D or 3D variants after the installation with:

```python
import softpool_cuda
from SoftPool import soft_pool1d, SoftPool1d
from SoftPool import soft_pool2d, SoftPool2d
from SoftPool import soft_pool3d, SoftPool3d
```

+ `soft_poolxd`: Is a functional interface for SoftPool.
+ `SoftPoolxd`: Is the class-based version which created an object that can be referenced later in the code.

## ImageNet models

All the ImageNet weights are available here: [link deducted for blind review - currently hosted on Google Drive]
## Citation

[To be updated]

## Licence

MIT
