"""
This file defines the core research contribution
"""
import os.path

import matplotlib
matplotlib.use('Agg')
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.encoders import psp_encoders
from models.stylegan2.model import Generator, EqualLinear
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pDSp(nn.Module):

	def __init__(self, opts):
		super(pDSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder().eval()
		self.decoder = Generator(self.opts.output_size, 512, 8).eval()
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		# # build a 3-layer projector
		# prev_dim = self.opts.n_styles*512
		# dim = 512
		# self.projector = nn.Sequential(nn.Linear(prev_dim, dim, bias=False),
		# 								nn.BatchNorm1d(dim),
		# 								nn.LeakyReLU(inplace=True), # first layer
		# 								nn.Linear(dim, dim, bias=False),
		# 								nn.BatchNorm1d(dim),
		# 								nn.LeakyReLU(inplace=True), # second layer
		# 								nn.Linear(dim, dim, bias=False),
		# 								nn.BatchNorm1d(dim, affine=False)) # output layer
		# # build a 2-layer predictor
		# pred_dim = 512
		# self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
		# 								nn.BatchNorm1d(pred_dim),
		# 								nn.LeakyReLU(inplace=True), # hidden layer
		# 								nn.Linear(pred_dim, pred_dim)) # output layer

		## change the scale of (||W_a|| + ||W_id||)
		# self.scale = nn.Parameter(torch.randn(1))

		# direction
		self.direction = nn.Parameter(torch.randn(self.opts.num_attributes, self.opts.n_styles*512))
		# self.direction_id_specific = nn.Sequential(
		# 	EqualLinear(512 * self.opts.n_styles + self.opts.num_attributes, 512 * self.opts.n_styles, lr_mul=1),
		# 	nn.ReLU(),
		# 	EqualLinear(512 * self.opts.n_styles, 512 * self.opts.n_styles, lr_mul=1),
		# 	nn.ReLU(),
		# 	EqualLinear(512 * self.opts.n_styles, 512 * self.opts.n_styles, lr_mul=1),
		# 	)
		# self.direction_alpha = nn.Parameter(torch.randn(1))
		# self.direction = nn.Parameter(torch.randn(self.opts.num_attributes, 512))
		# with torch.no_grad():
		# 	self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))  # .mul_(3)

		# Load weights if needed
		# self.load_interfaceGAN_direction()
		# with torch.no_grad():
		# 	self.direction.mul_(3)
		self.load_weights()

	def load_interfaceGAN_direction(self, dim=512):
		if self.opts.num_attributes == 5:
			model_path = "stylegan2_ffhq_boundary"
			attr_name = ['pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness']
		elif self.opts.num_attributes == 15:
			model_path = "stylegan2_ffhq_boundary_w_seperate_0.02"
			attr_name = ['pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness']

		boundary = np.zeros((self.opts.num_attributes, dim), dtype=np.float32)
		for i, attr in enumerate(attr_name):
			boundary_path = os.path.join(self.opts.pretrained_models_root,
										 f"{model_path}/boundary_{attr}.npy")
			boundary[i] = np.load(boundary_path)
		self.interfaceGAN = torch.from_numpy(boundary).to("cuda")
		# self.interfaceGAN = self.interfaceGAN.reshape(self.opts.num_attributes, dim)
		self.direction = nn.Parameter(self.interfaceGAN)

	def set_encoder(self):
		print("-------->",self.opts.encoder_type)
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'Encoder4Editing':
			encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'GradualStyleEncoder2Branch':
			encoder = psp_encoders.GradualStyleEncoder2Branch(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'EncodertoW2Branch':
			encoder = psp_encoders.EncodertoW2Branch(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'Encoder4Editing2Branch':
			encoder = psp_encoders.Encoder4Editing2Branch(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pDSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			# self.projector.load_state_dict(get_keys(ckpt, 'projector'), strict=True)
			# self.predictor.load_state_dict(get_keys(ckpt, 'predictor'), strict=True)
			self.__load_latent_avg(ckpt)
			self.__load_direction(ckpt)
		else:
			print('Loading encoders weights from e4e!')
			encoder_ckpt = torch.load(os.path.join(self.opts.pretrained_models_root, model_paths['e4e']))
			self.encoder.load_state_dict(encoder_ckpt["state_dict"], strict=True)
			# self.encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
			# print('Loading encoders weights from irse50!')
			# encoder_ckpt = torch.load(os.path.join(self.opts.pretrained_models_root, model_paths['ir_se50']))
			# # if input to encoder is not an RGB image, do not load the input layer weights
			# if self.opts.label_nc != 0:
			# 	encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			# self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(os.path.join(self.opts.pretrained_models_root, self.opts.stylegan_weights))
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, gt_attributes=None, edit_attributes=None, resize=True, latent_mask=None,
				input_code=False, randomize_noise=True, inject_latent=None, return_latents=False,
				alpha=None, return_disentangled_features=False):


		## generate editing attributes
		batch_size = x.shape[0]
		if edit_attributes is None:
			# edit_attributes = torch.randint(2, size=(batch_size, self.opts.num_attributes),
			# dtype=torch.float).to(self.opts.device)
			edit_attributes = torch.zeros(batch_size, self.opts.num_attributes)
			ind = torch.randint(self.opts.num_attributes, size=(batch_size, 1))
			for i in range(batch_size):
				if gt_attributes.size(1) > 0:
					if gt_attributes[i, ind[i]] == 0:
						edit_attributes[i, ind[i]] = 1
					else:
						edit_attributes[i, ind[i]] = -1
				else:
					edit_attributes[i, ind[i]] = 1
			edit_attributes = edit_attributes.to(self.opts.device)

		## generate latent vector w or w+
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# codes_specific_edit = self.direction_id_specific(
			# 	torch.cat((codes.detach().view(-1, self.opts.n_styles * 512), edit_attributes), 1))
			# codes_specific_edit = codes_specific_edit.view(-1, self.opts.n_styles, 512)

			# codes_a, codes_id = self.encoder(x)

			# normalize direction, codes_a, codes_id
			# with torch.no_grad():
			# 	self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))
			# codes_a.div_(torch.norm(codes_a, p=2, dim=(1,2), keepdim=True))
			# codes_id.div_(torch.norm(codes_id, p=2, dim=(1,2), keepdim=True))
			# print("dir",torch.norm(self.direction, dim=1))
			# codes_a = codes_a / (torch.norm(codes_a, p=2, dim=(2), keepdim=True))
			# codes_id = codes_id / (torch.norm(codes_id, p=2, dim=(2), keepdim=True))

			# codes = codes_a + codes_id

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
				# print("WE ARE HERE IN E4E CONDITION!")
				elif self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents)

		## generate editing image
		# codes_ori = torch.clone(codes).detach()
		codes_edit = torch.matmul(edit_attributes, self.direction)
		codes_edit = codes.detach() + codes_edit.reshape(batch_size, -1, 512)

		# codes_edit = torch.matmul(edit_attributes, self.direction).repeat(1, 18).reshape(batch_size, 18,
		# 																				 -1) + codes.detach()

		images_edit, _ = self.decoder([codes_edit],
									  input_is_latent=input_is_latent,
									  randomize_noise=randomize_noise,
									  return_latents=return_latents)

		if resize:
			images = self.face_pool(images)
			images_edit = self.face_pool(images_edit)

		# codes_a = codes_a.reshape(batch_size, -1)
		# codes_id = codes_id.reshape(batch_size, -1)
		codes_a = codes.reshape(batch_size, -1)
		codes_id = codes.reshape(batch_size, -1)
		return_codes = [(codes_a, codes_id)]
		if return_disentangled_features:
			D = torch.matmul(gt_attributes, self.direction)
			# D = self.direction_encoder(gt_attributes)

			# z1 = self.projector(codes_a.reshape(batch_size, -1))
			# z2 = self.projector(D)
			# p1 = self.predictor(z1)
			# p2 = self.predictor(z2)

			if self.opts.edit_cyc_lambda > 0:
				## cycle
				codes_cyc = self.encoder(images.detach())  # codes_a_cyc, codes_id_cyc
				return_codes.append(codes_cyc)

			if self.opts.edit_swap_lambda > 0:
				## swap
				codes_swaped_a_id = torch.zeros_like(codes_a)
				for i in range(batch_size):
					if i + 1 < batch_size:
						j = i + 1
					else:
						j = 0
					codes_swaped_a_id[i] = codes_a[i] + codes_id[j]
				if self.opts.start_from_latent_avg:
					if codes_swaped_a_id.ndim == 2:
						codes_swaped_a_id = codes_swaped_a_id + self.latent_avg.repeat(codes_swaped_a_id.shape[0], 1,
																					   1)[:, 0, :]
					elif self.opts.learn_in_w:
						codes_swaped_a_id = codes_swaped_a_id + self.latent_avg.repeat(codes_swaped_a_id.shape[0], 1)
					else:
						codes_swaped_a_id = codes_swaped_a_id + self.latent_avg.repeat(codes_swaped_a_id.shape[0], 1, 1)
				images_swaped_a_id, _ = self.decoder([codes_swaped_a_id],
													 input_is_latent=input_is_latent,
													 randomize_noise=randomize_noise,
													 return_latents=return_latents)
				codes_swap = self.encoder(images_swaped_a_id.detach())  # codes_a_cyc, codes_id_cyc
				return_codes.append(codes_swap)

			return images, return_codes, images_edit, edit_attributes, None

		# return images, result_latent, p1, p2, z1.detach(), z2.detach(), images_edit, codes_a, codes_id
		# return images, result_latent, codes_a, D, images_edit, codes_id, codes_cyc, codes_swap

		elif return_latents:
			return images, result_latent, images_edit, edit_attributes, None
		else:
			return images

	def forward_D(self, x, edit_attributes, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
				  inject_latent=None, return_latents=False, alpha=None, return_disentangled_features=False,
				  return_w_a=False):
		if input_code:
			codes = x
		else:
			codes_a = self.encoder(x)
			batch_size = codes_a.shape[0]
			## normalize direction, codes_a, codes_id
			# with torch.no_grad():
			# 	self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))
			# 	codes_a.div_(torch.norm(codes_a, p=2, dim=(1,2), keepdim=True))
			# 	codes_id.div_(torch.norm(codes_id, p=2, dim=(1,2), keepdim=True))
			# 	# print("dir",torch.norm(self.direction, dim=1))
			# import ipdb; ipdb.set_trace()
			# self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))
			# codes_a = codes_a / (torch.norm(codes_a, p=2, dim=(2), keepdim=True))
			# codes_id = codes_id / (torch.norm(codes_id, p=2, dim=(2), keepdim=True))
			# edit_attributes = torch.randint(2, size=(batch_size, self.opts.num_attributes),
			# 	dtype=torch.float).to(self.opts.device)
			# codes_edit = edit_attributes

			# codes_edit = torch.matmul(edit_attributes, self.interfaceGAN)
			# codes_edit = torch.matmul(edit_attributes, self.direction)
			# codes_edit = codes_edit.reshape(batch_size, -1, 512)
			codes_edit = edit_attributes.reshape(batch_size, -1, 512)

			# codes = self.scale *(codes_id )*1.0 + codes_edit.reshape(batch_size, -1, 512)
			# codes = self.scale *(codes_a + codes_id )+ codes_edit.reshape(batch_size, -1, 512)
			# codes = self.scale codes
			# codes = codes_a + codes_id + codes_edit.reshape(batch_size, -1, 512)

			## find the max of codes_edit
			# print(f"max position of direction: {torch.argmax(self.direction, dim=1, keepdim=True)}")
			# argmax = torch.argmax(self.direction, dim=1, keepdim=True)
			# argmax_v, indices =torch.topk(torch.abs(self.direction), 1, dim=1)
			# print(indices)

			# col_row = indices[torch.nonzero(edit_attributes[0])]
			codes_edit_new = torch.zeros_like(codes_a)
			# codes_edit_new[:, col_row//512, col_row%512] = codes_edit[:, col_row//512, col_row%512]
			# codes_edit_new[:, 6, 501] = 20 * codes_edit[:, 6, 501]
			codes_edit_new[:, 0 : 10, :] = codes_edit[:, 0 : 10, :]
			# codes_edit_new[:, 17, :] = codes_edit
			# print("------> 0 & 17")
			# codes_edit_new = codes_edit

			codes = codes_a + codes_edit_new
			# codes = codes_a + codes_edit
			# codes = codes_edit.reshape(batch_size, -1, 512)
			# print(self.scale)
			# print(self.direction)
			# print(codes_edit.size()) #torch.Size([1, 18, 512])
			# print(f"codes_a: [{codes_a.min()} ~ {codes_a.max()}], {codes_a.mean()}")
			# # print(f"codes_id: [{codes_id.min()} ~ {codes_id.max()}], {codes_id.mean()}")
			# print(f"codes_edit: [{codes_edit.min()} ~ {codes_edit.max()}], {codes_edit.mean()}")
			# print(f"codes: [{codes.min()} ~ {codes.max()}], {codes.mean()}")
			# import ipdb; ipdb.set_trace()

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
				# print("WE ARE HERE IN E4E CONDITION!")
				elif self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		# input_is_latent = input_code
		images, result_latent = self.decoder([codes],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents and return_disentangled_features:
			# projection
			codes_appearnce = codes_appearnce + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
			feat_enc = self.projection(codes_appearnce)
			feat_gt = self.projection(torch.matmul(gt_attributes, self.direction))
			# feat_gt = self.projection(torch.matmul(gt_attributes, self.interfaceGAN))

			# update direction
			# self.projection.eval()
			# feat_gt = self.projection(feat_gt_temp)
			# self.projection.train()
			# import ipdb; ipdb.set_trace()
			# print(self.direction) # .is_leaf

			return images, result_latent, feat_enc, feat_gt
		elif return_latents:
			return images, result_latent
		elif return_w_a:
			return images, codes_a.reshape(batch_size, -1)
		else:
			return images

	# def project(self, codes_appearnce, gt_attributes):
	# 	self.projection.eval()
	# 	codes = torch.clone(codes_appearnce).detach()
	# 	feat_enc = self.projection(codes)
	# 	feat_gt = self.projection(torch.matmul(gt_attributes, self.direction))
	# 	self.projection.train()

	# 	return feat_enc, feat_gt

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

	def __load_direction(self, ckpt):
		if 'direction' in ckpt:
			self.direction = nn.Parameter(ckpt['direction'])  # .to(self.opts.device)

# else:
# 	self.direction = None
# if 'scale' in ckpt:
# 	self.scale = nn.Parameter(ckpt['scale'])#.to(self.opts.device)
# else:
# 	self.scale = None



class pDSp_direction_512(nn.Module):

	def __init__(self, opts):
		super(pDSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder().eval()
		self.decoder = Generator(self.opts.output_size, 512, 8).eval()
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		# # build a 3-layer projector
		# prev_dim = self.opts.n_styles*512
		# dim = 512
		# self.projector = nn.Sequential(nn.Linear(prev_dim, dim, bias=False),
		# 								nn.BatchNorm1d(dim),
		# 								nn.LeakyReLU(inplace=True), # first layer
		# 								nn.Linear(dim, dim, bias=False),
		# 								nn.BatchNorm1d(dim),
		# 								nn.LeakyReLU(inplace=True), # second layer
		# 								nn.Linear(dim, dim, bias=False),
		# 								nn.BatchNorm1d(dim, affine=False)) # output layer
		# # build a 2-layer predictor
		# pred_dim = 512
		# self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
		# 								nn.BatchNorm1d(pred_dim),
		# 								nn.LeakyReLU(inplace=True), # hidden layer
		# 								nn.Linear(pred_dim, pred_dim)) # output layer

		## change the scale of (||W_a|| + ||W_id||)
		# self.scale = nn.Parameter(torch.randn(1))

		# direction
		# self.direction = nn.Parameter(torch.randn(self.opts.num_attributes, self.opts.n_styles*512))
		self.direction = nn.Parameter(torch.randn(self.opts.num_attributes, 512))
		with torch.no_grad():
			self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))#.mul_(3)

		# Load weights if needed
		# self.load_interfaceGAN_direction()
		# with torch.no_grad():
		# 	self.direction.mul_(3)
		self.load_weights()

	def load_interfaceGAN_direction(self, dim=512):
		if self.opts.num_attributes == 5:
			model_path = "stylegan2_ffhq_boundary"
			attr_name = ['pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness']
		elif self.opts.num_attributes == 15:
			model_path = "stylegan2_ffhq_boundary_w_seperate_0.02"
			attr_name = ['pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness']
		
		boundary = np.zeros((self.opts.num_attributes, dim), dtype=np.float32)
		for i, attr in enumerate(attr_name):
			boundary_path = os.path.join(self.opts.pretrained_models_root, 
										f"{model_path}/boundary_{attr}.npy")
			boundary[i] = np.load(boundary_path)
		self.interfaceGAN = torch.from_numpy(boundary).to("cuda")
		# self.interfaceGAN = self.interfaceGAN.reshape(self.opts.num_attributes, dim)
		self.direction = nn.Parameter(self.interfaceGAN)

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'Encoder4Editing':
			encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'GradualStyleEncoder2Branch':
			encoder = psp_encoders.GradualStyleEncoder2Branch(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'EncodertoW2Branch':
			encoder = psp_encoders.EncodertoW2Branch(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'Encoder4Editing2Branch':
			encoder = psp_encoders.Encoder4Editing2Branch(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pDSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			# self.projector.load_state_dict(get_keys(ckpt, 'projector'), strict=True)
			# self.predictor.load_state_dict(get_keys(ckpt, 'predictor'), strict=True)
			self.__load_latent_avg(ckpt)
			self.__load_direction(ckpt)
		else:
			print('Loading encoders weights from e4e!')
			encoder_ckpt = torch.load(os.path.join(self.opts.pretrained_models_root, model_paths['e4e']))
			self.encoder.load_state_dict(encoder_ckpt["state_dict"], strict=True)
			# self.encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
			# print('Loading encoders weights from irse50!')
			# encoder_ckpt = torch.load(os.path.join(self.opts.pretrained_models_root, model_paths['ir_se50']))
			# # if input to encoder is not an RGB image, do not load the input layer weights
			# if self.opts.label_nc != 0:
			# 	encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			# self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(os.path.join(self.opts.pretrained_models_root, self.opts.stylegan_weights))
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, gt_attributes=None, edit_attributes=None, resize=True, latent_mask=None,
				input_code=False, randomize_noise=True, inject_latent=None, return_latents=False, 
	            alpha=None, return_disentangled_features=False):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# codes_a, codes_id = self.encoder(x)

			# normalize direction, codes_a, codes_id
			# with torch.no_grad():
			# 	self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))
				# codes_a.div_(torch.norm(codes_a, p=2, dim=(1,2), keepdim=True))
				# codes_id.div_(torch.norm(codes_id, p=2, dim=(1,2), keepdim=True))
			# print("dir",torch.norm(self.direction, dim=1))
			# codes_a = codes_a / (torch.norm(codes_a, p=2, dim=(2), keepdim=True))
			# codes_id = codes_id / (torch.norm(codes_id, p=2, dim=(2), keepdim=True))

			# codes = codes_a + codes_id

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
					# print("WE ARE HERE IN E4E CONDITION!")
				elif self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)
		batch_size = codes.shape[0]
		# batch_size = codes_a.shape[0]

		## generate editing image
		if edit_attributes is None:
			# edit_attributes = torch.randint(2, size=(batch_size, self.opts.num_attributes), 
				# dtype=torch.float).to(self.opts.device)
			edit_attributes = torch.zeros(batch_size, self.opts.num_attributes)
			ind = torch.randint(self.opts.num_attributes, size=(batch_size, 1))
			for i in range(batch_size):
				if gt_attributes.size(1) > 0:
					if gt_attributes[i, ind[i]] == 0:
						edit_attributes[i, ind[i]] = 1
					else:
						edit_attributes[i, ind[i]] = -1
				else:
					edit_attributes[i, ind[i]] = 1
			edit_attributes = edit_attributes.to(self.opts.device)

		# codes_ori = torch.clone(codes).detach()
		# codes_edit = torch.matmul(edit_attributes, self.direction) + codes.detach()
		# codes_edit = codes.detach() + codes_edit.reshape(batch_size, -1, 512)

		codes_edit = torch.matmul(edit_attributes, self.direction).repeat(1,18).reshape(batch_size,18,-1) + codes.detach()

		images_edit, _ = self.decoder([codes_edit],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)
			images_edit = self.face_pool(images_edit)

		# codes_a = codes_a.reshape(batch_size, -1)
		# codes_id = codes_id.reshape(batch_size, -1)
		codes_a = codes.reshape(batch_size, -1)
		codes_id = codes.reshape(batch_size, -1)
		return_codes = [(codes_a, codes_id)]
		if return_disentangled_features:
			D = torch.matmul(gt_attributes, self.direction)
			# D = self.direction_encoder(gt_attributes)

			# z1 = self.projector(codes_a.reshape(batch_size, -1))
			# z2 = self.projector(D)
			# p1 = self.predictor(z1)
			# p2 = self.predictor(z2)

			if self.opts.edit_cyc_lambda > 0:
				## cycle
				codes_cyc = self.encoder(images.detach()) # codes_a_cyc, codes_id_cyc
				return_codes.append(codes_cyc)

			if self.opts.edit_swap_lambda > 0:
				## swap
				codes_swaped_a_id=torch.zeros_like(codes_a)
				for i in range(batch_size):
					if i + 1 < batch_size:
						j = i + 1
					else:
						j = 0
					codes_swaped_a_id[i] = codes_a[i] + codes_id[j]
				if self.opts.start_from_latent_avg:
					if codes_swaped_a_id.ndim == 2:
						codes_swaped_a_id = codes_swaped_a_id + self.latent_avg.repeat(codes_swaped_a_id.shape[0], 1, 1)[:, 0, :]
					elif self.opts.learn_in_w:
						codes_swaped_a_id = codes_swaped_a_id + self.latent_avg.repeat(codes_swaped_a_id.shape[0], 1)
					else:
						codes_swaped_a_id = codes_swaped_a_id + self.latent_avg.repeat(codes_swaped_a_id.shape[0], 1, 1)
				images_swaped_a_id, _ = self.decoder([codes_swaped_a_id],
				                                     input_is_latent=input_is_latent,
				                                     randomize_noise=randomize_noise,
				                                     return_latents=return_latents)
				codes_swap = self.encoder(images_swaped_a_id.detach()) # codes_a_cyc, codes_id_cyc
				return_codes.append(codes_swap)

			return images, return_codes, D, images_edit, edit_attributes
			# return images, result_latent, p1, p2, z1.detach(), z2.detach(), images_edit, codes_a, codes_id
			# return images, result_latent, codes_a, D, images_edit, codes_id, codes_cyc, codes_swap

		elif return_latents:
			return images, result_latent, images_edit, edit_attributes
		else:
			return images


	def forward_D(self, x, edit_attributes, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None, return_disentangled_features=False,
	            return_w_a=False):
		if input_code:
			codes = x
		else:
			codes_a = self.encoder(x)
			batch_size = codes_a.shape[0]
			## normalize direction, codes_a, codes_id
			# with torch.no_grad():
			# 	self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))
			# 	codes_a.div_(torch.norm(codes_a, p=2, dim=(1,2), keepdim=True))
			# 	codes_id.div_(torch.norm(codes_id, p=2, dim=(1,2), keepdim=True))
			# 	# print("dir",torch.norm(self.direction, dim=1))
			# import ipdb; ipdb.set_trace()
			# self.direction.div_(torch.norm(self.direction, p=2, dim=1, keepdim=True))
			# codes_a = codes_a / (torch.norm(codes_a, p=2, dim=(2), keepdim=True))
			# codes_id = codes_id / (torch.norm(codes_id, p=2, dim=(2), keepdim=True))
			# edit_attributes = torch.randint(2, size=(batch_size, self.opts.num_attributes), 
			# 	dtype=torch.float).to(self.opts.device)
			# codes_edit = edit_attributes

			# codes_edit = torch.matmul(edit_attributes, self.interfaceGAN)
			codes_edit = torch.matmul(edit_attributes, self.direction)
			# codes_edit = codes_edit.reshape(codes_a.shape[0], 512)

			# codes = self.scale *(codes_id )*1.0 + codes_edit.reshape(batch_size, -1, 512)
			# codes = self.scale *(codes_a + codes_id )+ codes_edit.reshape(batch_size, -1, 512)
			# codes = self.scale codes 
			# codes = codes_a + codes_id + codes_edit.reshape(batch_size, -1, 512)

			# codes_edit_new = torch.zeros_like(codes_a)
			# codes_edit_new[:, 0,:] = codes_edit
			# codes_edit_new[:, 17, :] = codes_edit
			# print("------> 0 & 17")
			codes_edit_new = codes_edit

			codes = codes_a + codes_edit_new
			# codes = codes_a + codes_edit
			# codes = codes_edit.reshape(batch_size, -1, 512)
			# print(self.scale)
			# print(self.direction)
			# print(codes_edit.size()) #torch.Size([1, 18, 512])
			print(f"codes_a: [{codes_a.min()} ~ {codes_a.max()}], {codes_a.mean()}")
			# print(f"codes_id: [{codes_id.min()} ~ {codes_id.max()}], {codes_id.mean()}")
			print(f"codes_edit: [{codes_edit.min()} ~ {codes_edit.max()}], {codes_edit.mean()}")
			print(f"codes: [{codes.min()} ~ {codes.max()}], {codes.mean()}")
			# import ipdb; ipdb.set_trace()

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
					# print("WE ARE HERE IN E4E CONDITION!")
				elif self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		# input_is_latent = input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents and return_disentangled_features:
			# projection
			codes_appearnce = codes_appearnce + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
			feat_enc = self.projection(codes_appearnce)
			feat_gt = self.projection(torch.matmul(gt_attributes, self.direction))
			# feat_gt = self.projection(torch.matmul(gt_attributes, self.interfaceGAN))
			
			# update direction
			# self.projection.eval()
			# feat_gt = self.projection(feat_gt_temp)
			# self.projection.train()
			# import ipdb; ipdb.set_trace()
			# print(self.direction) # .is_leaf

			return images, result_latent, feat_enc, feat_gt
		elif return_latents:
			return images, result_latent
		elif return_w_a:
			return images, codes_a.reshape(batch_size, -1)
		else:
			return images

	# def project(self, codes_appearnce, gt_attributes):
	# 	self.projection.eval()
	# 	codes = torch.clone(codes_appearnce).detach()
	# 	feat_enc = self.projection(codes)
	# 	feat_gt = self.projection(torch.matmul(gt_attributes, self.direction))
	# 	self.projection.train()

	# 	return feat_enc, feat_gt


	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

	def __load_direction(self, ckpt):
		if 'direction' in ckpt:
			self.direction = nn.Parameter(ckpt['direction'])#.to(self.opts.device)
		# else:
		# 	self.direction = None
		# if 'scale' in ckpt:
		# 	self.scale = nn.Parameter(ckpt['scale'])#.to(self.opts.device)
		# else:
		# 	self.scale = None



