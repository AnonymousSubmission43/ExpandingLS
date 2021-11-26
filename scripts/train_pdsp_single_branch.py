"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_pdsp_single_branch import Coach


def main():
	opts = TrainOptions().parse()
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	opts.exp_dir = "../experiments/pdsp_wClassfier"
	opts.pretrained_models_root = "../pretrained_models"
	opts.dataset_type = "ffhq_encode"
	opts.datasets_root = "../../Dataset/FaceRepresentation/"
	opts.checkpoint_path = "../pretrained_models/psp_ffhq_encode.pt"
	opts.return_img_name = True
	opts.start_from_latent_avg = True

	opts.seed = 43
	opts.stylegan_size = 1024
	opts.output_size = 1024
	opts.dual_direction = False 
	if opts.dataset_type == "ffhq_encode":
		opts.num_attributes = 5
		opts.attr_path = "ffhq/attributes_smallset.json"
	elif opts.dataset_type == "celeba_encode":
		opts.num_attributes = 18 #40
		opts.attr_path = "celeba_hq/attributes_celeba.json"
	opts.return_attributes = True
	opts.workers = 4
	opts.batch_size = 4
	opts.test_batch_size = 2
	opts.test_workers = 2
	# opts.encoder_type = "GradualStyleEncoder"
	opts.encoder_type = "GradualStyleContentEncoder"
	opts.style_2d = True
	opts.train_encoder = True 
	opts.train_decoder = True
	opts.direction_2d = True

	opts.l2_lambda = 0.01
	opts.id_lambda = 1
	opts.lpips_lambda = 0.8
	opts.pred_lambda = 10

	opts.edit_pred_lambda = 10
	opts.edit_id_lambda = 0
	opts.edit_lpips_lambda = 0
	opts.histogram_lambda = 0
	opts.entropy_lambda = 0
	opts.edit_l2_lambda = 0.1
	opts.feat2d_norm_lambda = 0
	opts.align_lambda = 1
	opts.tv_lambda = 0.0001

	opts.edit_cyc_lambda = 0 
	opts.edit_adv_lambda = 0
	opts.w_norm_lambda = 0
	opts.orth_lambda = 0
	opts.sim_lambda = 0
	opts.edit_swap_lambda = 0

	if not os.path.exists(opts.exp_dir):
		# raise Exception('Oops... {} already exists'.format(opts.exp_dir))
		os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
