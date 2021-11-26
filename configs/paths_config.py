import os
datasets_path = '' # '/mnt/H/data/'
dataset_paths = {
	'celeba_train': os.path.join(datasets_path, 'celeba_hq/train/all'),
	'celeba_test': os.path.join(datasets_path, 'celeba_hq/val/all'),
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': os.path.join(datasets_path, 'ffhq/resized256'),
	'celeba_hq': os.path.join(datasets_path, 'celeba_hq'),
	'ffhq256': os.path.join(datasets_path, 'ffhq/resized256'),
}

pretrained_models_path = '' #'/mnt/H/pretrained_models/'
model_paths = {
	'stylegan_ffhq': os.path.join(pretrained_models_path, 'stylegan2-ffhq-config-f.pt'),
	'e4e': os.path.join(pretrained_models_path, 'e4e_ffhq_encode.pt'),
	'ir_se50': os.path.join(pretrained_models_path, 'model_ir_se50.pth'),
	'circular_face': os.path.join(pretrained_models_path, 'CurricularFace_Backbone.pth'),
	'mtcnn_pnet': os.path.join(pretrained_models_path,'mtcnn/pnet.npy'),
	'mtcnn_rnet': os.path.join(pretrained_models_path, 'mtcnn/rnet.npy'),
	'mtcnn_onet': os.path.join(pretrained_models_path, 'mtcnn/onet.npy'),
	'shape_predictor': os.path.join(pretrained_models_path, 'shape_predictor_68_face_landmarks.dat'),
	'moco': os.path.join(pretrained_models_path, 'moco_v2_800ep_pretrain.pth.tar')
}
