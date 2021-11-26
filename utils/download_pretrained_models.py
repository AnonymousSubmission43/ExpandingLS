import os.path
import sys
from utils.data_utils import download_from_google_drive
# from configs.paths_config import pretrained_models_path

sys.path.append(".")
sys.path.append("..")

pretrained_models_path = '/mnt/H/pretrained_models'
file_id_name = [('1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT', 'stylegan2-ffhq-config-f.pt'),
                ('1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn', 'model_ir_se50.pth'),
                ('1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja', 'mtcnn.tar.gz'),
                ('1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj', 'CurricularFace_Backbone.pth'),
                ('1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO', '550000.pt'),
                ('18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe', 'moco_v2_800ep_pretrain.pt'),
                ('1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0', 'psp_ffhq_encode.pt'),
                ('1_S4THAzXb-97DbpXmanjHtXRyKxqjARv', 'psp_ffhq_frontalization.pt'),
                ('1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA', 'psp_celebs_sketch_to_face.pt'),
                ('1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz', 'psp_celebs_seg_to_face.pt'),
                ('1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu', 'psp_celebs_super_resolution.pt'),
                ('1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz', 'psp_ffhq_toonify.pt'),
                ("1cUv_reLE6k3604or78EranS7XzuVMWeO", "e4e_ffhq_encode.pt"),
                ("17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV", "e4e_cars_encode.pt"),
                ("1TkLLnuX86B_BMo2ocYD0kX9kWh53rUVX", "e4e_horse_encode.pt"),
                ("1-L0ZdnQLwtdy6-A_Ccgq5uNJGTqE7qBa", "e4e_church_encode.pt")
                ]


for file_id, file_name in file_id_name:
    if not os.path.isfile(os.path.join(pretrained_models_path, file_name)):
        download_from_google_drive(file_id, file_name, pretrained_models_path)



