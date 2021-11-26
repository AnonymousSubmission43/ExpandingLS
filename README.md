# ExpandingLS: Expanding the Latent Space of StyleGAN for Real Face Editing

### Installation
- Clone this repo:
``` 
git clone https://github.com/eladrich/pixel2style2pixel.git
cd pixel2style2pixel
```
- Dependencies:
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/).
All dependencies for defining the environment are provided in `environment/psp_env.yaml`.


### Preparing your Data
- Currently, we provide support for numerous datasets and experiments (encoding, frontalization, etc.).
    - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation.
    - Refer to `configs/transforms_config.py` for the transforms defined for each dataset/experiment.
    - Finally, refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
      as well as the transforms.
As an example, assume we wish to run encoding using ffhq (`dataset_type=ffhq_encode`).
We first go to `configs/paths_config.py` and define:
``` 
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```


#### **Pretrained Models**
https://drive.google.com/drive/folders/1_P_jnP3ZyTlVR2RX1NjvA2Gy2sRWaJlw?usp=sharing


#### **Training**
```
python scripts/train_pdsp_single_branch.py \
--dataset_type=ffhq_encode \
--exp_dir=/path/to/experiment \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1
```

#### **Training**
```
python scripts/edit_pdsp_styleGAN_single_branch.py \
--exp_path=../experiments/test \
--pretrained_models_path=../pretrained_models/pretrained/iteration_60000.pt \
--sample_celeba_test=../../Dataset/FaceRepresentation/celeba_hq/val/all/ \
--save_gt --save_rec --evaluate
```
