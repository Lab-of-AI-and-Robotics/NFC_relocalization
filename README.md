# NFC_relocalization

### **This is a prototype version and will be updated to a more user-friendly version for future execution.**

NFC_relocalization is Lidar-base Large-scale Global Place Recognition and Relocalization method by estimate 6-DOF transform

### Enviroment setup

NFC_relocalization is based on [LCDnet]. If you want to use NFC_relocalization, you can use LCDnet's docker or this way

1. Install [PyTorch](https://pytorch.org/) (make sure to select the correct cuda version)
2. Install the requirements ```pip install -r requirements.txt```
3. Install [spconv](https://github.com/traveller59/spconv) <= 2.1.25 (make sure to select the correct cuda version, for example ```pip install spconv-cu113==2.1.25``` for cuda 11.3)
4. Install [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
5. Install [faiss-cpu](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) - NOTE: avoid installing faiss via pip, use the conda version, or build it from source alternatively.

## Preprocessing

Download the [KITTI](http://semantic-kitti.org/dataset.html#download) dataset and Preprocessing for your training:

```
python -m data_process.generate_loop_GT_KITTI --root_folder KITTI_ROOT
```

KITTI_ROOT is your download place the KITTI dataset



### Ground Plane Removal

You can use Ground Plane Removal remove_ground_plane_kitti oin data_process. However, for better results, we recommend using [patchwork++](https://github.com/url-kaist/patchwork-plusplus).

### Training
The training script is not support to parallel GPU learning. During training, you can only use one GPU.

To train on the KITTI dataset:
```
python -m training_KITTI_DDP --root_folder KITTI_ROOT --dataset kitti --batch_size B --without_ground
```


## Acknowledgements
This implementation is based on [LCDnet]

[LCDnet]: https://github.com/robot-learning-freiburg/LCDNet
