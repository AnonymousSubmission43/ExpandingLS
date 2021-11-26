import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils import data

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss, adversarial, attribute_prediction, histogram_loss, align_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.pdsp_single_branch import pDSp
from training.ranger import Ranger
import time
import numpy as np
# from azureml.core.run import Run
import random
from tqdm import tqdm
from datetime import datetime
import warnings
import pprint
import json


ATTRIBUTES = ['pose', 'glass', 'age', 'gender', 'smile']
# # ATTRIBUTES2 = ['pose', '-pose', 'glass', '-glass', 'age', '-age', 'gender', '-gender', 'smile', '-smile']
# ATTRIBUTES_all = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
# 			  'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
# 			  'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
# 			  'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
# 			  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
# 			  'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
# 			  'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
# 			  'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
# selected_attrs_ind = [4, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23, 26, 29, 30, 31, 36]
# ATTRIBUTES = [ATTRIBUTES_all[i] for i in selected_attrs_ind]
# # attributes are: ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Chubby',
# # 				 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
# # 				 'Male', 'Mustache', 'Narrow_Eyes', 'Pale_Skin', 'Rosy_Cheeks',
# # 				 'Sideburns', 'Smiling', 'Wearing_Lipstick']

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Coach:
	def __init__(self, opts):
		self.opts = opts
		self.global_step = 0
		# self.run = Run.get_context()

		# self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		# self.opts.device = self.device

		# Allow multiple GPU
		n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
		self.distributed = n_gpu > 1
		self.device = "cuda"
		self.opts.device = self.device
		self.local_rank = opts.local_rank

		if self.distributed:
			torch.cuda.set_device(opts.local_rank)
			torch.distributed.init_process_group(backend="nccl", init_method="env://")
			synchronize()
		set_seed(self.opts.seed)

		if get_rank() == 0:
			print('Number of GPUs: ', n_gpu)
			dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + f'_GPU{opts.local_rank}' 
			self.opts.exp_dir = os.path.join(self.opts.exp_dir, dt_string)
			if os.path.exists(self.opts.exp_dir):
				# raise Exception('Oops... {} already exists'.format(opts.exp_dir))
				warnings.warn('Note ... {} already exists'.format(self.opts.exp_dir))
			else:
				os.makedirs(self.opts.exp_dir)

			opts_dict = vars(opts)
			pprint.pprint(opts_dict)
			with open(os.path.join(self.opts.exp_dir, 'opt.json'), 'w') as f:
				json.dump(opts_dict, f, indent=4, sort_keys=True)

		# Initialize network
		self.net = pDSp(self.opts).to(self.device)

		if self.distributed:
			self.net = nn.parallel.DistributedDataParallel(
				self.net,
				device_ids=[opts.local_rank],
				output_device=opts.local_rank,
				broadcast_buffers=False,
				find_unused_parameters=True,
			)
			self.net_module = self.net.module
		else:
			self.net_module = self.net

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net_module.latent_avg is None:
			self.net_module.latent_avg = self.net_module.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0 or self.opts.edit_lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex', pretrained_model_path=opts.pretrained_models_root).to(self.device).eval()
		if self.opts.id_lambda > 0 or self.opts.edit_id_lambda > 0:
			self.id_loss = id_loss.IDLoss(pretrained_model_path=opts.pretrained_models_root).to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()
		if self.opts.sim_lambda > 0:
			self.sim_loss = nn.CosineSimilarity(dim=1).to(self.device)
		if self.opts.edit_adv_lambda > 0:
			self.edit_adv_loss = adversarial.Adversarial().to(self.device)
		if self.opts.edit_pred_lambda > 0 or self.opts.pred_lambda > 0:
			if self.opts.num_attributes == 5:
				self.attr_pred_loss = attribute_prediction.AttrPredLoss_5(
										pretrained_model_path=opts.pretrained_models_root,
										dual_direction=opts.dual_direction).to(self.device).eval()
			elif self.opts.num_attributes in [18, 40]:
				self.attr_pred_loss = attribute_prediction.AttrPredLoss_40(
										pretrained_model_path=opts.pretrained_models_root,
										dual_direction=opts.dual_direction).to(self.device).eval()
		if self.opts.histogram_lambda > 0:
			self.histogram_loss = histogram_loss.HistogramLoss(self.device).to(self.device).eval()
		if self.opts.align_lambda > 0:
			self.align_loss = align_loss.AlignLoss().to(self.device).eval()
		# if self.opts.entropy_lambda > 0:
		# 	self.entropy_loss = histogram_loss.HistogramLoss(self.device).to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		# self.optimizer_direction = self.configure_optimizers_direction()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   sampler=data_sampler(self.train_dataset, shuffle=True, distributed=self.distributed),
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  sampler=data_sampler(self.test_dataset, shuffle=False, distributed=self.distributed),
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(self.opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		self.train_run_times = []
		self.train_log_times = []
		self.val_run_log_times = []
		if not self.opts.dual_direction:
			self.ATTRIBUTES = ATTRIBUTES
		else:
			self.ATTRIBUTES = ATTRIBUTES2


	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			train_run_time = 0
			train_log_time = 0
			val_run_time = 0
			for batch_idx, batch in enumerate(self.train_dataloader):
				tic = time.time()
				self.optimizer.zero_grad()
				# self.optimizer_direction.zero_grad()
				x, y, x_name, y_name, gt_attributes = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				gt_attributes = gt_attributes.to(self.device).float()
				## ori framework
				# y_hat, latents, y_hat_edit, edit_attr = self.net.forward(
				# 	x, gt_attributes, return_disentangled_features=True)
				# loss, loss_dict, id_logs = self.calc_loss(
				# 	x, y, y_hat, y_hat_edit, latents, edit_attr, gt_attributes=gt_attributes)
				y_hat, latents, y_hat_edit, edit_attr, rec2 = self.net.forward(
					x, gt_attributes, return_disentangled_features=True)
				loss, loss_dict, id_logs = self.calc_loss(
					x, y, y_hat, y_hat_edit, rec2, latents, edit_attr, gt_attributes=gt_attributes)
				loss.backward()
				self.optimizer.step()
				# self.optimizer_direction.step()

				toc = time.time()
				self.train_run_times.append(toc - tic)

				# Logging related
				if get_rank() == 0 and (self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 50 == 0)):
					tic = time.time()
					for i in range(len(id_logs)):
						# self.parse_and_log_images([id_logs[i]], x[[i]], y[[i]], y_hat[[i]], title='images/train/faces',
						attr_ind = torch.nonzero(edit_attr[i], as_tuple=True)[0]
						self.parse_and_log_images([id_logs[i]], x[[i]], y_hat_edit[[i]], y_hat[[i]], title='images/train/faces',
												  subscript=f"{x_name[i]}_{self.ATTRIBUTES[attr_ind]}",
												  display_count=1)
					toc = time.time()
					self.train_log_times.append(toc - tic)
				if get_rank() == 0 and self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train') 
					print(f"learned direction: {self.net_module.direction}")

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if get_rank() == 0 and (self.global_step % self.opts.save_interval == 0
										or self.global_step == self.opts.max_steps):
					tic = time.time()
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)
					toc = time.time()
					train_log_time += toc - tic

				if get_rank() == 0 and self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1


	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			tic = time.time()
			x, y, x_name, y_name, gt_attributes = batch

			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				gt_attributes = gt_attributes.to(self.device).float()
				## ori framework
				# y_hat, latent, y_hat_edit, edit_attr = self.net.forward(x, gt_attributes,
				# 	return_latents=True, return_disentangled_features=False)
				# loss, cur_loss_dict, id_logs = self.calc_loss(
				# 	x, y, y_hat, y_hat_edit, val=True)
				y_hat, latent, y_hat_edit, edit_attr, rec2 = self.net.forward(x, gt_attributes,
					return_latents=True, return_disentangled_features=False)
				loss, cur_loss_dict, id_logs = self.calc_loss(
					x, y, y_hat, y_hat_edit, rec2, val=True)

			agg_loss_dict.append(cur_loss_dict)				

			# Logging related
			if self.global_step % (self.opts.save_image_per_val_interval * self.opts.val_interval) == 0 or self.global_step == self.opts.max_steps:
				for i in range(len(id_logs)):
					attr_ind = torch.nonzero(edit_attr[i], as_tuple=True)[0]
					if x_name[i] in self.visualization_img_list:
						# self.parse_and_log_images([id_logs[i]], x[[i]], y[[i]], y_hat[[i]],
						self.parse_and_log_images([id_logs[i]], x[[i]], y_hat_edit[[i]], y_hat[[i]],
												  title='images/test/faces',
												  subscript=f"{x_name[i]}_{self.ATTRIBUTES[attr_ind]}",#'{:04d}'.format(batch_idx),
												  display_count=1)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch
			toc = time.time()
			self.val_run_log_times.append(toc - tic)
		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		if get_rank() == 0:
			self.log_metrics(loss_dict, prefix='test')
			self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		# params = []
		params = [self.net_module.direction]
		# params += list(self.net_module.direction_id_specific.parameters())
		# params += [self.net_module.direction_alpha]
		if self.opts.train_encoder:
			params += list(self.net_module.encoder.contents.parameters())
			params += list(self.net_module.encoder.latlayer3.parameters())
			params += list(self.net_module.encoder.latlayer4.parameters())
			# params += list(self.net_module.encoder.parameters())
			# params += list(self.net_module.encoder.styles.parameters())
			# params += list(self.net_module.encoder.latlayer1.parameters())
			# params += list(self.net_module.encoder.latlayer2.parameters())

		if self.opts.train_decoder:
			params += list(self.net_module.decoder.fuse.parameters())
			# params += list(self.net_module.decoder.attention_fuse.parameters())
			# params += list(self.net_module.decoder.to_rgbs.parameters())
			# params += list(self.net_module.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer
	
	# def configure_optimizers_direction(self):
	# 	params = [self.net_module.direction]
	# 	if self.opts.optim_name == 'adam':
	# 		optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_direction)
	# 	else:
	# 		optimizer = Ranger(params, lr=self.opts.learning_rate_direction)
	# 	return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		self.visualization_img_list = [fname.split('.')[0] for fname in test_dataset.source_fnames[:self.opts.num_visualization_img]]
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, y_hat_edit=None, rec2=None, latents=[], edit_attr=[],
		 		  gt_attributes=[], val=False):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if  (val == False) and self.opts.w_norm_lambda > 0:
			latent = latents[0]
			loss_w_norm = self.w_norm_loss(latent, self.net_module.latent_avg[:latent.shape[1], :])
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if  (val == False) and self.opts.feat2d_norm_lambda > 0:
			latent_2d = latents[1]
			loss_2d_feat_norm = 0
			# loss_2d_feat_norm = torch.sum(latent_2d.norm(2, dim=(1, 2))) / latent_2d.shape[0]
			for feat in latent_2d:
				loss_2d_feat_norm += torch.mean(feat.norm(2, dim=(1,2,3)))
			loss_dict['loss_2d_feat_norm'] = float(loss_2d_feat_norm)
			loss += loss_2d_feat_norm * self.opts.feat2d_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda

		if (val == False) and self.opts.pred_lambda > 0:
			# edit attributes predicion loss
			loss_pred = self.attr_pred_loss(y_hat, gt_attributes+edit_attr)
			loss_dict['loss_pred'] = float(loss_pred)
			loss += loss_pred * self.opts.pred_lambda


		## editing loss:
		if self.opts.align_lambda > 0:
			# I_edit: y_hat, I_edit_refine: y_hat_edit, I_rec: rec2 or rec2[0], I_rec_refine: rec2[1]
			# diff = (y_hat_edit - rec2[1].detach()) - (y_hat.detach() - rec2[0].detach())

			loss_align = self.align_loss(y_hat_edit, x, y_hat, rec2)
			loss_dict['loss_align'] = float(loss_align)
			loss += loss_align * self.opts.align_lambda
		if self.opts.tv_lambda > 0:
			loss_tv = torch.mean(torch.abs(y_hat_edit[:,:,:-1,:] - y_hat_edit[:,:,1:,:])) \
					  + torch.mean(torch.abs(y_hat_edit[:,:,:,:-1] - y_hat_edit[:,:,:,1:]))
			loss_dict['loss_tv'] = float(loss_tv)
			loss += loss_tv * self.opts.tv_lambda
		if self.opts.entropy_lambda > 0:
			# alpha = 3.5
			## l2 loss
			# D_norms = self.net_module.direction.norm(2, dim=(1))
			## l1 loss
			D_norms = self.net_module.direction[:, : self.opts.n_styles*512].norm(1, dim=(1))
			# for D_norm in D_norms:
			# 	if D_norm < alpha:
			# 		D_norm = D_norm * 0
			## exclude pose
			if self.opts.num_attributes == 5:
				# loss_entropy = torch.sum(D_norms[1:]) / (self.net_module.direction.size(0) - 1)
				loss_entropy = (torch.sum(D_norms[1:]) + 0.1 * torch.sum(D_norms[0])) / self.net_module.direction.size(0)
			else:
				loss_entropy = torch.sum(D_norms) / self.net_module.direction.size(0)
			# loss_entropy = torch.sum(D_norms) / self.net_module.direction.size(0)
			loss_dict['loss_entropy'] = float(loss_entropy)
			loss += loss_entropy * self.opts.entropy_lambda
		if (val == False) and self.opts.histogram_lambda > 0:
			# loss_histogram = self.histogram_loss(y_hat, y_hat_edit)
			# loss_histogram = self.histogram_loss(x, y_hat)
			loss_histogram = self.histogram_loss(x, y_hat_edit)
			loss_dict['loss_histogram'] = float(loss_histogram)
			loss += loss_histogram * self.opts.histogram_lambda
		if (val == False) and self.opts.edit_pred_lambda > 0:
			# edit attributes predicion loss
			## check if classifer match with gt
			# loss_edit_pred = self.attr_pred_loss(x, gt_attributes)
			## compare image: y_hat_edit and x
			loss_edit_pred = self.attr_pred_loss.comp_img(x, y_hat_edit, edit_attr)
			# ## compare y_hat_edit and gt label
			# loss_edit_pred = self.attr_pred_loss(y_hat_edit, gt_attributes+edit_attr)
			loss_dict['loss_edit_pred'] = float(loss_edit_pred)
			loss += loss_edit_pred * self.opts.edit_pred_lambda
		if (val == False) and  self.opts.edit_adv_lambda > 0:
			loss_edit_adv = self.edit_adv_loss(y_hat_edit, x)
			loss_dict['loss_edit_adv'] = float(loss_edit_adv)
			loss += loss_edit_adv * self.opts.edit_adv_lambda
		if self.opts.edit_id_lambda > 0:
			loss_edit_id, _, _ = self.id_loss(y_hat_edit, x.detach(), y)
			# loss_edit_id, _, _ = self.id_loss(y_hat_edit, y_hat.detach(), y)
			loss_dict['loss_edit_id'] = float(loss_edit_id)
			loss += loss_edit_id * self.opts.edit_id_lambda
		if self.opts.edit_lpips_lambda > 0:
			loss_edit_lpips = self.lpips_loss(y_hat_edit, x.detach())
			# loss_edit_lpips = self.lpips_loss(y_hat_edit, y_hat.detach())
			loss_dict['loss_edit_lpips'] = float(loss_edit_lpips)
			loss += loss_edit_lpips * self.opts.edit_lpips_lambda
		if self.opts.edit_l2_lambda > 0:
			loss_edit_l2 = F.mse_loss(y_hat_edit, x)
			loss_dict['loss_edit_l2'] = float(loss_edit_l2)
			loss += loss_edit_l2 * self.opts.edit_l2_lambda

		if (val == False) and self.opts.edit_cyc_lambda > 0:
			loss_edit_cyc = F.mse_loss(latents, x.detach())
			loss_dict['loss_cyc'] = float(loss_edit_cyc)
			loss += loss_edit_cyc * self.opts.edit_cyc_lambda


		"""
		if (val == False):
			if self.opts.orth_lambda > 0:
				## orthogal loss on directions
				norm_d = self.net_module.direction / (torch.norm(
					self.net_module.direction, p=2, dim=1, keepdim=True))
				D_Dt = torch.mm(norm_d, norm_d.t()) 
				# D_Dt = torch.mm(self.net_module.direction, self.net_module.direction.t()) 
				loss_D_orth = torch.norm(D_Dt - torch.eye(self.opts.num_attributes).cuda())
				loss_dict['loss_D_orth'] = float(loss_D_orth)
				loss += loss_D_orth * self.opts.orth_lambda

			w_a = latents[0][0]
			w_id = latents[0][1]
			if (val == False) and self.opts.sim_lambda > 0:
				# edit similarity loss
				loss_d = 0
				for i, d_i in enumerate(self.net_module.direction):
					d_i = d_i.repeat(w_a.size(0), 1)
					s_i = gt_attributes[:, i]
					loss_d += (((self.sim_loss(d_i, w_a) + 1.0) / 2.0 - s_i)**2).mean()
				loss_dict['loss_edit_sim'] = float(loss_d)
				loss += loss_d * self.opts.sim_lambda

				# loss_disentangle_Wa_D = (self.sim_loss(d1, w)**2 + self.sim_loss(p2, z1).mean()) * 0.5
				# loss_dict['loss_disentangle_Wa_D'] = float(loss_disentangle_Wa_D)
				# loss += loss_disentangle_Wa_D * self.opts.sim_lambda

				# w_a = w_a.view(w_a.shape[0], -1)
				# w_id = w_id.view(w_id.shape[0], -1)
				# loss_disentangle_Wa_Wid = torch.abs(self.sim_loss(w_a, w_id)).mean() 
				# loss_dict['loss_disentangle_Wa_Wid'] = float(loss_disentangle_Wa_Wid)
				# loss += loss_disentangle_Wa_Wid * self.opts.sim_lambda

			code_ind = 1
			if (val == False) and self.opts.edit_cyc_lambda > 0:
				codes_cyc = latents[code_ind]
				code_ind += 1
				batch_size = w_a.shape[0]
				loss_edit_cyc_att = F.mse_loss(w_a.detach(), codes_cyc[0].reshape(batch_size, -1))
				loss_edit_cyc_id = F.mse_loss(w_id.detach(), codes_cyc[1].reshape(batch_size, -1))
				loss_dict['loss_cyc_attr'] = float(loss_edit_cyc_att)
				loss_dict['loss_cyc_id'] = float(loss_edit_cyc_id)
				loss += (loss_edit_cyc_id + loss_edit_cyc_att) * self.opts.edit_cyc_lambda

			if (val == False) and self.opts.edit_swap_lambda > 0:
				batch_size = w_a.shape[0]
				codes_swap = latents[code_ind]
				w_swap_a = codes_swap[0]
				w_swap_id = codes_swap[1]
				loss_edit_swap_id = 0
				loss_edit_swap_attr = 0
				for i in range(batch_size):
					if i + 1 < batch_size:
						j = i + 1
					else:
						j = 0
					loss_edit_swap_attr += 1 - w_swap_a.detach()[i].dot(y_feats[i])
					loss_edit_swap_id += 1 - w_swap_id.detach()[i].dot(y_feats[j])
					# loss_edit_swap_id += F.mse_loss(w_id[j].detach(), codes_swap[1].reshape(batch_size, -1)[i])
				loss_edit_swap_attr = float(loss_edit_swap_attr) / batch_size
				loss_edit_swap_id = float(loss_edit_swap_id) / batch_size

				# loss_edit_swap_attr = F.mse_loss(w_a.detach(), codes_swap[0].reshape(batch_size, -1))
				loss_dict['loss_swap_id'] = loss_edit_swap_id
				loss_dict['loss_swap_attr'] = float(loss_edit_swap_attr)
				loss += (loss_edit_swap_id + loss_edit_swap_attr) * self.opts.edit_swap_lambda
		"""
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	# def calc_loss_direction(self, feat_enc, feat_gt, loss_dict, val=False):
	# 	loss = 0.0		
	# 	if val == False:
	# 		loss_disentangle = self.disentangle_loss(feat_enc, feat_gt)
	# 		loss_dict['loss_direction'] = float(loss_disentangle)
	# 		loss += loss_disentangle * self.opts.sim_lambda
	# 	return loss, loss_dict

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)
			# self.run.log('{}/{}'.format(prefix, key), value)

	def print_metrics(self, metrics_dict, prefix):
		message = 'Metrics for {}, step={}'.format(prefix, self.global_step)
		# print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			message += '\t{}={:.4f}  '.format(key, value)
			# print('\t{} = '.format(key), value)

		if self.global_step > 0:
			message += '\t train_run_time {:.4f}+-{:.4f}'.format(np.mean(self.train_run_times), np.std(self.train_run_times))
			message += '\t train_log_time {:.4f}+-{:.4f}'.format(np.mean(self.train_log_times), np.std(self.train_log_times))
			message += '\t val_run_time {:.4f}+-{:.4f}'.format(np.mean(self.val_run_log_times), np.std(self.val_run_log_times))
		print(message)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net_module.state_dict(),
			'opts': vars(self.opts),
			'direction': self.net_module.direction,
			# 'scale': self.net_module.scale
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net_module.latent_avg
		return save_dict
