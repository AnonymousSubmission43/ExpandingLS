import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim

class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)

def update_opts_namespace(opts, updates):
	opts_dict = vars(opts)
	for key, val in updates.items():
		opts_dict[key] = val
	return Bunch(opts_dict)

# Log images
def log_input_image(x, opts):
	if opts.label_nc == 0:
		return tensor2im(x)
	elif opts.label_nc == 1:
		return tensor2sketch(x)
	else:
		return tensor2map(x)


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def tensor2map(var):
	mask = np.argmax(var.data.cpu().numpy(), axis=0)
	colors = get_colors()
	mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
	for class_idx in np.unique(mask):
		mask_image[mask == class_idx] = colors[class_idx]
	mask_image = mask_image.astype('uint8')
	return Image.fromarray(mask_image)


def tensor2sketch(var):
	im = var[0].cpu().detach().numpy()
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	im = (im * 255).astype(np.uint8)
	return Image.fromarray(im)


# Visualization utils
def get_colors():
	# currently support up to 19 classes (for the celebs-hq-mask dataset)
	colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
			  [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
			  [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
	return colors


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Edited\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Reconstruction\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Edited')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Reconstruction')



def quantize(img, rgb_range):
	pixel_range = 255 / rgb_range
	return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def convert_rgb_to_y(tensor):
	image = tensor[0].cpu().numpy().transpose(1,2,0)#.detach()

	if len(image.shape) <= 2 or image.shape[2] == 1:
		return image

	#xform = np.array([[65.481, 128.553, 24.966]])
	#y_image = image.dot(xform.T) + 16.0

	xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
	y_image = image.dot(xform.T) + 16.0

	return y_image

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
	# Y channel
	if hr.nelement() == 1: return 0

	# diff = (sr - hr) / rgb_range
	# shave = scale
	# if diff.size(1) > 1:
	#     gray_coeffs = [65.738, 129.057, 25.064]
	#     convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
	#     diff = diff.mul(convert).sum(dim=1)

	# valid = diff[..., shave:-shave, shave:-shave]
	# mse = valid.pow(2).mean()

	shave = scale
	image1 = convert_rgb_to_y(sr)
	image2 = convert_rgb_to_y(hr)
	image1 = image1[shave:-shave, shave:-shave, :]
	image2 = image2[shave:-shave, shave:-shave, :]
	psnr = compare_psnr(image1, image2, data_range=rgb_range)
	ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
						sigma=1.5, data_range=rgb_range)

	# return ssim
	return psnr, ssim