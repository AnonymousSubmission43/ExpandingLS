# ExpandingLS: Expanding the Latent Space of StyleGAN for Real Face Editing

### Installation
- Clone this repo:
``` 
git clone https://github.com/AnonymousSubmission43/ExpandingLS.git
cd ExpandingLS
```
- Dependencies:
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/).
All dependencies for defining the environment are provided in `environment/py3pt160.yaml`.


### Preparing your Data
- Currently, we provide support for CelebA-HQ and FFHQ datasets.
    - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation.
    - Refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
      as well as the transforms.
For example, we first go to `configs/paths_config.py` and define:
``` 
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```

#### **Pretrained Models**\
Please download all the pretrained model, and put it in the folder of '../pretrained_models'.
https://drive.google.com/drive/folders/1_P_jnP3ZyTlVR2RX1NjvA2Gy2sRWaJlw?usp=sharing


#### **Training**
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 scripts/train_pdsp_single_branch.py \
--dataset_type=ffhq_encode \
--exp_dir=/path/to/experiment \
--start_from_latent_avg 
```

#### **Training**
```
python scripts/edit_pdsp_styleGAN_single_branch.py \
--exp_path=../experiments/test \
--pretrained_models_path=../pretrained_models/pretrained/iteration_60000.pt \
--sample_celeba_test=../../Dataset/FaceRepresentation/celeba_hq/val/all/ \
--save_gt --save_rec --evaluate
```
