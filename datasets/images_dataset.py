import os.path

import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import json

class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(os.path.join(opts.datasets_root, source_root)))
		self.source_fnames = sorted(os.listdir(os.path.join(opts.datasets_root, source_root)))
		self.target_paths = sorted(data_utils.make_dataset(os.path.join(opts.datasets_root, target_root)))
		self.target_fnames = sorted(os.listdir(os.path.join(opts.datasets_root, target_root)))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

		f = open(os.path.join(self.opts.datasets_root, self.opts.attr_path), )
		self.attribute_data = json.load(f)
		if self.opts.dataset_type == "ffhq_encode":
			self.select_attributes = list(self.attribute_data.keys())[1:]
			if opts.dual_direction:
				self.num_of_attributes = self.num_of_attributes * 2
			# data: dict_keys(['img', 'pose_Yaw', 'glass_NoGlasses', 'glass_HasGlasses', 'age',
			# 	'facialHair_Beard', 'facialHair_Moustache', 'facialHair_Sideburns', 'gender',
			# 	'hairColor_Black', 'hairColor_Brown', 'hairColor_Blond', 'hairColor_Gray',
			# 	'hairColor_Red', 'eyeMakeup_NoMakeup', 'eyeMakeup_HasMakeup', 'lipMakeup_NoMakeup',
			# 	'lipMakeup_HasMakeup', 'emotion_Happiness', 'emotion_Neutral'])
			# small set: ['pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness']

			# if self.opts.dataset_type == "ffhq_encode":
			# 	self.exc = ".txt"
			# elif self.opts.dataset_type == "celeba_encode":
			# 	self.exc = ".jpg"
		elif self.opts.dataset_type == "celeba_encode":
			self.all_attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
			  'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
			  'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
			  'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
			  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
			  'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
			  'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
			  'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
			self.selected_attrs_ind = [4, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23, 26, 29, 30, 31, 36]
			self.select_attributes = [self.all_attributes[i] for i in self.selected_attrs_ind]

		print(f"attributes are: {self.select_attributes}")
		self.num_of_attributes = len(self.select_attributes)
		assert self.num_of_attributes == self.opts.num_attributes, "Num of attributes is not set correctly."

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		# index = 12
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		if self.opts.return_img_name and self.opts.return_attributes:
			img = self.source_fnames[index].split('.')[0]
			try:
				attributes = []
				if self.opts.dataset_type == "ffhq_encode":
					idx = self.attribute_data['img'].index(f"{img}.txt")
					for attr in self.select_attributes:
						attributes.append(self.attribute_data[attr][idx])
						if self.opts.dual_direction:
							attributes.append(1 - self.attribute_data[attr][idx])
				elif self.opts.dataset_type == "celeba_encode":
					attributes_all = self.attribute_data[f"{img}.jpg"]
					attributes = [attributes_all[i] for i in self.selected_attrs_ind]
					# print(f"---------->{img}")
					# print(attributes)
					# self.selected_attrs_ind = [4, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23, 26, 29, 30, 31, 36]
					# self.select_attributes = [self.all_attributes[i]


			except:
				attributes = []
			attributes = torch.Tensor(attributes)
			# attributes.div_(torch.norm(attributes, p=2, dim=0, keepdim=True))	

			# print(attributes.size())
			# print(torch.norm(attributes, dim=0))
			# print(attributes)
			# print(img)
			# import ipdb; ipdb.set_trace()

			return from_im, to_im, img, self.target_fnames[index].split('.')[0], attributes
		elif self.opts.return_img_name:
			return from_im, to_im, self.source_fnames[index].split('.')[0], self.target_fnames[index].split('.')[0]
		else:
			return from_im, to_im
