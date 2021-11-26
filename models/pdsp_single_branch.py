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

import cv2
import numpy as np
def plot_feat_map(x, save_path='./'):
    """
        x.size: batch_size * channel * w * h
    """
    for imgs in x:
        total_channels = imgs.size(0)
        if total_channels == 3:
            # RGB
            imgs = imgs.cpu().numpy().transpose(1,2,0)
            imgs = ((imgs + 1) / 2)
            imgs[imgs < 0] = 0
            imgs[imgs > 1] = 1
            imgs = np.ascontiguousarray(imgs)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

            if(imgs.max() <= 1):
                imgs = (imgs * 255).astype(np.uint8)

            # cv2.imshow(save_path,imgs)
            cv2.imwrite(save_path, imgs)
            # cv2.waitKey(0)
        else:
            axis = np.floor(np.sqrt(total_channels))
            x = 0
            for i, gray in enumerate(imgs):
                # C feature maps
                gray = gray.cpu().numpy()
                gray = np.ascontiguousarray(gray)
                gray_no_bb = gray[1:-1, 1:-1]
                gray = (gray - gray_no_bb.min())/(gray_no_bb.max() - gray_no_bb.min()) * 255
                gray = cv2.applyColorMap(gray.astype(np.uint8) , cv2.COLORMAP_JET)
                w, h, _ = gray.shape
                # plot each row
                y = i % axis
                if y == 0:
                    row = gray
                else:
                    row = np.concatenate((row, gray), axis=1)
                # add row to final figure
                if (i+1) // axis != x:
                    # if the final row is shorter than others, concat blank
                    if row.shape[1] < axis * w:
                        blank = np.zeros((int(h), int(axis * w - row.shape[1]), 3))
                        row = np.concatenate((row, blank), axis=1)
                    # init row or concat new raw
                    if x == 0:
                        add_row = row
                    else:
                        add_row = np.concatenate((add_row, row), axis=0)
                    x = x + 1
            # cv2.imshow(save_path,add_row)
            cv2.imwrite(save_path,add_row)
            # cv2.waitKey(0)

    # normalized = imgs.mul(255)
    # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    # imageio.imwrite(save_path, tensor_cpu.numpy())
def plot_feat_map_sum(x, save_path='./'):
    """
        x.size: batch_size * channel * w * h
    """
    for imgs in x:
        total_channels = imgs.size(0)
        if total_channels == 3:
            # RGB
            plot_feat_map(x, save_path)
        else:
            gray = imgs.cpu().numpy()
            gray = np.ascontiguousarray(gray)
            gray = np.sum(gray, 0)
            gray_no_bb = gray[1:-1, 1:-1]
            gray = (gray - gray_no_bb.min())/(gray_no_bb.max() - gray_no_bb.min()) * 255
            gray = cv2.applyColorMap(gray.astype(np.uint8) , cv2.COLORMAP_JET)
            # cv2.imshow(save_path, gray)
            cv2.imwrite(save_path, gray)
            # cv2.waitKey(0)


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

		# direction
		if self.opts.direction_2d:
			self.opts.n_styles_2d = self.opts.n_styles // 2
			self.direction = nn.Parameter(torch.randn(self.opts.num_attributes,
													  (self.opts.n_styles+self.opts.n_styles_2d)*512))
		else:
			self.direction = nn.Parameter(torch.randn(self.opts.num_attributes, self.opts.n_styles*512))


		self.load_weights()

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
		elif self.opts.encoder_type == 'GradualStyleContentEncoder':
			encoder = psp_encoders.GradualStyleContentEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'Encoder4ContentEditing':
			encoder = psp_encoders.Encoder4ContentEditing(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		# print(encoder)
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pDSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=False)
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
				input_code=False, randomize_noise=False, inject_latent=None, return_latents=False,
				alpha=None, return_disentangled_features=False):
		if input_code:
			codes = x
		else:
			if self.opts.style_2d:
				codes, feat_2d = self.encoder(x)
			else:
				codes = self.encoder(x)

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
				# print("WE ARE HERE IN E4E CONDITION!")
				elif self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, :codes.shape[1], :]

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
		batch_size = codes.shape[0]

		## generate editing image
		if edit_attributes is None:
			edit_attributes = torch.zeros(batch_size, self.opts.num_attributes)
			if not self.opts.dual_direction:
				ind = torch.randint(self.opts.num_attributes, size=(batch_size, 1))
				for i in range(batch_size):
					if gt_attributes.size(1) > 0:
						if gt_attributes[i, ind[i]] == 0:
							edit_attributes[i, ind[i]] = 1
						else:
							edit_attributes[i, ind[i]] = -1
					else:
						edit_attributes[i, ind[i]] = 1
			else:
				for i in range(batch_size):
					if gt_attributes.size(1) > 0:
						ind = torch.nonzero((gt_attributes == 0), as_tuple=True)
						ind_ind = torch.randint(self.opts.num_attributes//2, size=(batch_size, 1))
						edit_attributes[ind[0][ind_ind[i]+self.opts.num_attributes*(i-1)], ind[1][ind_ind[i]+self.opts.num_attributes*(i-1)]] = 1
						if ind[1][ind_ind[i]+self.opts.num_attributes*(i-1)] % 2 == 0:
							edit_attributes[ind[0][ind_ind[i]+self.opts.num_attributes*(i-1)], ind[1][ind_ind[i]+self.opts.num_attributes*(i-1)] + 1] = -1
						else:
							edit_attributes[ind[0][ind_ind[i]+self.opts.num_attributes*(i-1)], ind[1][ind_ind[i]+self.opts.num_attributes*(i-1)] - 1] = -1
					else:
						ind = torch.randint(self.opts.num_attributes, size=(batch_size, 1))
						edit_attributes[i, ind[i]] = 1
						if ind[i] % 2 == 0:
							edit_attributes[i, ind[i] + 1] = -1
						else:
							edit_attributes[i, ind[i] - 1] = -1
			edit_attributes = edit_attributes.to(self.opts.device)

		codes_attr = torch.matmul(edit_attributes, self.direction[:, :self.opts.n_styles*512].detach()) * 1.5
		codes_edit = codes.detach() + codes_attr.reshape(batch_size, -1, 512)

		## generate direction for 2d features
		code_edit_2d = torch.matmul(1 - torch.abs(edit_attributes), self.direction[:, self.opts.n_styles*512:])
		code_edit_2d = code_edit_2d.reshape(batch_size, -1, 512)

		with torch.no_grad():
			## W
			images_rec, feat_rec = self.decoder([codes.detach()],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
								   			 return_features=True)
								   			 # return_latents=return_latents)

		## W + Direction
		images_edit, feat_edit = self.decoder([codes_edit],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
								   			 return_features=True)
								   			 # return_latents=return_latents)

		codes_edit2 = codes_attr.clone().detach()
		codes_edit2 = codes.detach() + codes_edit2.reshape(batch_size, -1, 512)
		## W + 2D features + Direction + Direction_2d
		images_edit_refine, _ = self.decoder([codes_edit2],
									  input_is_latent=input_is_latent,
									  randomize_noise=randomize_noise,
									  return_latents=return_latents,
									  struct=feat_2d,
									  edit_2d=code_edit_2d)

		# ## W + 2D features + Direction + Direction_2d
		# images_edit_refine, _ = self.decoder([codes_edit2],
		# 							  input_is_latent=input_is_latent,
		# 							  randomize_noise=randomize_noise,
		# 							  return_latents=return_latents,
		# 							  struct=feat_2d,
		# 							  edit_2d=code_edit_2d,
		# 							  attention=[feat_rec, feat_edit])


		if resize:
			images_edit = self.face_pool(images_edit) # edit: W + D
			images_edit_refine = self.face_pool(images_edit_refine) # edit_refine: W + D + 2d feat

			images_rec2 = self.face_pool(images_rec)# rec: W

		return_codes = []
		if return_disentangled_features:

			images_edit_refine_cyc = None
			if self.opts.edit_cyc_lambda > 0:
				## cycle
				# codes_cyc = self.encoder(images.detach())  # codes_a_cyc, codes_id_cyc
				# return_codes.append(codes_cyc)
				codes_edit_refine, feat_2d_edit_refine = self.encoder(images_edit_refine.detach())
				codes_cyc = codes_edit_refine.detach() - codes_edit.clone().detach() \
							+ self.latent_avg.repeat(codes_edit_refine.shape[0], 1, 1)[:, :codes_edit_refine.shape[1], :]

				# codes_cyc = codes_cyc.detach() - codes_edit3.reshape(batch_size, -1, 512)

				images_edit_refine_cyc, _ = self.decoder([codes_cyc],
											  input_is_latent=input_is_latent,
											  randomize_noise=randomize_noise,
											  return_latents=return_latents,
											  struct=feat_2d_edit_refine,
											  edit_2d=code_edit_2d,
											  attention=[feat_rec, feat_edit])
				if resize:
					images_edit_refine_cyc = self.face_pool(images_edit_refine_cyc) # images_edit_refine_cyc

				# return_codes.append(images_edit_refine_cyc)

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


			return images_edit, images_edit_refine_cyc, images_edit_refine, edit_attributes, images_rec2
			# return images_edit, return_codes, images_edit_refine, edit_attributes, images_rec2
			# return images, return_codes, images_edit, edit_attributes


		elif return_latents:
			return images_edit, return_codes, images_edit_refine, edit_attributes, images_rec2
		else:
			return images_edit

	def forward_D(self, x, edit_attributes, resize=True, latent_mask=None, input_code=False, randomize_noise=False,
				  inject_latent=None, return_latents=False, alpha=None, return_disentangled_features=False,
				  return_w_a=False):
		if input_code:
			codes_a = x
		else:
			if self.opts.style_2d:
				codes_a, feat_2d = self.encoder(x)
			else:
				codes_a = self.encoder(x)
				# codes_a, _ = self.encoder(x)
				# feat_2d = None
			# feat_2d = F.interpolate(feat_2d, size=4) * 100
			# feat_2d = nn.Parameter(torch.randn(1, 512, 4, 4).cuda())
			# noise = torch.zeros(1,512,4,4).cuda()
			# feat_2d[:,: ,0,0] = codes_a.reshape(batch_size, ) [:,17,:]
			# feat_2d[:,: ,3,3] = noise[:,: ,3,3]
			# feat_2d[:,: ,2,2] = codes_a[:,17,:]
			# feat_2d[:,: ,3,3] = codes_a[:,17,:]

			# codes_a, feat_2d = self.encoder(x)
			batch_size = codes_a.shape[0]

			# # codes_edit = torch.matmul(edit_attributes, self.interfaceGAN)
			# codes_edit = edit_attributes.reshape(batch_size, -1, 512)
			# codes_edit = torch.matmul(edit_attributes, self.direction)
			# codes_edit = codes_edit.reshape(batch_size, -1, 512)

			# codes_edit = edit_attributes[:, :, :self.opts.n_styles*512].reshape(batch_size, -1, 512)
			codes_edit = edit_attributes[:, :self.opts.n_styles*512].reshape(batch_size, -1, 512)

			## find the max of codes_edit
			# print(f"max position of direction: {torch.argmax(self.direction, dim=1, keepdim=True)}")
			# argmax = torch.argmax(self.direction, dim=1, keepdim=True)
			# argmax_v, indices =torch.topk(torch.abs(self.direction), 1, dim=1)
			# print(indices)

			# col_row = indices[torch.nonzero(edit_attributes[0])]
			# codes_edit_new = torch.zeros_like(codes_a)
			# codes_edit_new[:, col_row//512, col_row%512] = codes_edit[:, col_row//512, col_row%512]
			# codes_edit_new[:, 0 : 10, :] = codes_edit[:, 0 : 10, :]
			# codes_edit_new[:, 17, :] = codes_edit
			# print("------> 0 & 17")
			# codes_edit_new = codes_edit
			# codes_edit_new[:, 4, 501] += 20 #codes_edit[:, 4, 501] + 10
			codes = codes_a + codes_edit
			# codes = codes_a + codes_edit
			# codes = codes_edit.reshape(batch_size, -1, 512)
			# print(self.scale)
			# print(self.direction)
			# # print(codes_edit.size()) #torch.Size([1, 18, 512])
			# print(f"codes_a: [{codes_a.min()} ~ {codes_a.max()}], {codes_a.mean()}")
			# # print(f"codes_id: [{codes_id.min()} ~ {codes_id.max()}], {codes_id.mean()}")
			# print(f"codes_edit: [{codes_edit.min()} ~ {codes_edit.max()}], {codes_edit.mean()}")
			# print(f"codes: [{codes.min()} ~ {codes.max()}], {codes.mean()}")
			# # import ipdb; ipdb.set_trace()

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
				# print("WE ARE HERE IN E4E CONDITION!")
				elif self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, :codes.shape[1], :]

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

		# ## W + 2D features + Direction + Direction_2d
		# code_edit_2d = torch.matmul(edit_attributes, self.direction[:, self.opts.n_styles*512:])
		# code_edit_2d = code_edit_2d.reshape(batch_size, -1, 512)
		if self.opts.style_2d:
			if not self.opts.direction_2d:
				images, result_latent = self.decoder([codes],
													 input_is_latent=input_is_latent,
													 randomize_noise=randomize_noise,
													 return_latents=return_latents,
													 struct=feat_2d)
			else:
				code_edit_2d = edit_attributes[:, self.opts.n_styles*512:].reshape(batch_size, -1, 512)

				# # W + 2D features + Direction + Direction_2d
				# images, result_latent = self.decoder([codes],
				# 							  input_is_latent=input_is_latent,
				# 							  randomize_noise=randomize_noise,
				# 							  return_latents=return_latents,
				# 							  struct=feat_2d,
				# 							  edit_2d=code_edit_2d)

				# """ with attention """
				# with torch.no_grad():
				# 	## W
				# 	_, feat_rec = self.decoder([codes - codes_edit],
				# 							 input_is_latent=input_is_latent,
				# 							 randomize_noise=randomize_noise,
				# 							 return_features=True)
				# 	## W + Direction
				# 	_, feat_edit = self.decoder([codes],
				# 							 input_is_latent=input_is_latent,
				# 							 randomize_noise=randomize_noise,
				# 							 return_features=True)
				images, result_latent = self.decoder([codes],
											  input_is_latent=input_is_latent,
											  randomize_noise=randomize_noise,
											  return_latents=return_latents,
											  struct=feat_2d,
											  edit_2d=code_edit_2d,)
											  # attention=[feat_rec, feat_edit])

		else:
			images, result_latent = self.decoder([codes],
												 input_is_latent=input_is_latent,
												 randomize_noise=randomize_noise,
												 return_latents=return_latents,)
		if resize:
			images = self.face_pool(images)

		if return_latents and return_disentangled_features:
			# projection
			codes_appearnce = codes_appearnce + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
			feat_enc = self.projection(codes_appearnce)
			feat_gt = self.projection(torch.matmul(gt_attributes, self.direction))

			return images, result_latent, feat_enc, feat_gt
		elif return_latents:
			return images, result_latent
		elif return_w_a:
			return images, codes_a.reshape(batch_size, -1)
		else:
			return images



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
			print("Direction size: ", ckpt['direction'].size())
			if self.opts.direction_2d:
				self.opts.n_styles_2d = self.opts.n_styles // 2
				if ckpt['direction'].size(1) < (self.opts.n_styles + self.opts.n_styles_2d) * 512:
					direction = torch.cat([ckpt['direction'], torch.randn(self.opts.num_attributes, self.opts.n_styles_2d*512)], 1)
					self.direction = nn.Parameter(direction)
				else:
					self.direction = nn.Parameter(ckpt['direction'])
			else:
				self.direction = nn.Parameter(ckpt['direction'])  # .to(self.opts.device)



