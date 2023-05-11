# SyMFM6D: Symmetry-aware Multi-directional Fusion for Multi-View 6D Object Pose Estimation

This is the official source code for the paper **"SyMFM6D: Symmetry-aware Multi-directional Fusion for Multi-View 
6D Object Pose Estimation"** by Fabian Duffhauss et al. 
The code allows the users to reproduce and extend the results reported in the study. 
Please cite the paper when reporting, reproducing, or extending the results.

## Installation
- Install CUDA 10.1
- Create a virtual environment with all required packages:
    ```shell script
    conda create -n SyMFM6D python=3.6
    conda activate SyMFM6D
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
    conda install -c anaconda -c conda-forge scikit-learn
    conda install pytorch-lightning==1.4.9 torchmetrics==0.6.0 -c conda-forge
    conda install matplotlib einops tensorboardx pandas opencv==3.4.2 -c conda-forge
    pip install opencv-contrib-python==3.4.2.16
    ```

- Compile the [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) operators:
    ```shell script
    cd models/RandLA/
    sh compile_op.sh
    ```

- Download and install [normalSpeed](https://github.com/hfutcgncas/normalSpeed):
    ```shell script
    git clone https://github.com/hfutcgncas/normalSpeed.git
    cd normalSpeed/normalSpeed
    python3 setup.py install
    ```

## Datasets and Models
- The YCB-Video dataset ca be downloaded 
[here](https://drive.google.com/file/d/1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi/view?usp=sharing).
- The MV-YCB SymMovCam dataset can be downloaded 
[here](https://drive.google.com/file/d/16p0keTKr_UQnu7wHS8AgFIFe1GGS1qet/view?usp=share_link). 
Using this dataset requires the 3D models of the YCB-Video dataset which can be downloaded
[here](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing).

After downloading a new dataset, the zip file needs to be extracted and the paths to the 3D models and the datasets
need to be modified in [common.py](common.py). For training a new model from scratch, 
a ResNet-34 pre-trained on ImageNet is required which can be downloaded 
[here](https://download.pytorch.org/models/resnet34-333f7ec4.pth). 


## Finding Object Symmetries
Our symmetry-aware training procedure requires the rotational symmetry axes of all objects. We compute them once in
advance of the training by running the script [find_symmetries.py](utils/find_symmetries.py) for each object. For
objects with multiple symmetry axes, the script needs to be ran multiple times with different initial values for the 
symmetry axis, e.g.
```shell script
python utils/find_symmetries.py --obj_name 024_bowl --symmtype rotational
```

We also provide pre-computed symmetry axes for all objects which we used for the paper in 
[symmetries.txt](datasets/ycb/dataset_config/symmetries.txt).


## Training Models

In the following, we give a few examples how to train models with our SyMFM6D approach on the different datasets.


### Single-View Training on YCB-Video
```shell script
python run.py --dataset ycb --workers 4 --run_name YcbVideo_1view_training --epochs 40 --batch_size 9 \
    --sift_fps_kps --symmetry 1 --n_rot_sym 16
```

### Multi-View Training on YCB-Video
```shell script
python run.py --dataset ycb --workers 4 --run_name YcbVideo_3views_training --epochs 10 --batch_size 3 \
    --sift_fps_kps --symmetry 1 --n_rot_sym 16 --multi_view 1 --set_views 3  --checkpoint single_view_checkpoint.ckpt \
    --lr_scheduler reduce --lr 7e-05
```

### Single-View Training on MV-YCB SymMovCam
```shell script
python run.py --dataset SymMovCam --workers 4 --run_name MvYcbSymMovCam_1view_training --epochs 60 \
    --batch_size 3 --sift_fps_kps --symmetry 1 --n_rot_sym 16
```

### Multi-View Training on MV-YCB SymMovCam
```shell script
python run.py --dataset SymMovCam --workers 4 --run_name MvYcbSymMovCam_1view_training --epochs 60 \
    --batch_size 3 --sift_fps_kps --symmetry 1 --n_rot_sym 16 --multi_view 1 --set_views 3
```


## Evaluating Models

A model can be evaluated by specifying a checkpoint using `--checkpoint <name_of_checkpoint>` and by adding the 
argument `--test`, e.g.
```shell script
python run.py --dataset ycb --workers 4 --run_name YcbVideo_3views_evaluation --batch_size 3 --sift_fps_kps \
    --multi_view --set_views 3  --checkpoint YcbVideo_3views_checkpoint.ckpt 
```

