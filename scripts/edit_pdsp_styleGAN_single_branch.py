import os
import argparse
from argparse import Namespace
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import json

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image, calc_psnr
from options.test_options import TestOptions
from models.pdsp_single_branch import pDSp
import torchvision.transforms as transforms
from utils import common, train_utils
import matplotlib.pyplot as plt
from utils import data_utils
from editings import latent_editor

from criteria.lpips.lpips import LPIPS
from models.mtcnn.mtcnn import MTCNN
from models.encoders.model_irse import IR_101
import json

def run_seperate_enc_dir(args):
    exp_path = os.path.join(args.exp_path, args.direction_model)
    os.makedirs(exp_path, exist_ok=True)
    save_results = open(os.path.join(exp_path, "results.txt"), "w")

    if args.direction_model in ['SeFa', 'ganspace']:
        exp_type = args.direction_model
    else:
        exp_type = 'edit_single'
    print(f"Edit Method: {args.direction_model}")
    ##---------------- Load data
    f = open('../../Dataset/FaceRepresentation/ffhq/attributes_smallset.json')
    attribute_data = json.load(f)
    select_attributes = list(attribute_data.keys())[1:]
    print("attributes list: ", select_attributes)

    ##----------------  load encoder model
    if args.enc_model == "mine":
        model_path = args.pretrained_models_path
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']

    elif args.enc_model == "e4e":
        model_path = f"../pretrained_models/{args.enc_model}_ffhq_encode.pt"
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['encoder_type'] = "Encoder4Editing"
    elif args.enc_model == "psp":
        model_path = f"../pretrained_models/{args.enc_model}_ffhq_encode.pt"
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['encoder_type'] = "GradualStyleEncoder"
    print("model path: ", model_path)
    opts['pretrained_models_root'] = "../pretrained_models"
    opts['checkpoint_path'] = model_path
    opts['num_attributes'] = 5
    # opts['stylegan_size'] = 1024
    opts['device'] = "cuda"

    if 'dual_direction' not in opts.keys():
        opts['dual_direction'] = False
    if 'direction_2d' not in opts.keys():
        opts['direction_2d'] = False
    if 'style_2d' not in opts.keys():
        opts['style_2d'] = False

    opts = Namespace(**opts)
    opts.output_size = 1024
    opts.learn_in_w = True

    ##---------------- Settings
    if exp_type == "edit_multi":
        attributes_idxs = [3, 4]
        attributes_save_names = f'{select_attributes[attributes_idxs[0]]}_{select_attributes[attributes_idxs[1]]}'

    exp_path_gt = args.exp_path_gt

    ##----------------  load direction from another model
    if args.direction_model == 'mine':
        if args.enc_model == 'mine':
            model_path2 = model_path
        else:
            model_path2 = args.pretrained_models_path
        ckpt2 = torch.load(model_path2, map_location='cpu')
        direction2 = ckpt2['direction']

    elif args.direction_model == 'InterfaceGAN':
        attr_names = ['pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness']
        boundary = np.zeros((opts.num_attributes, 512), dtype=np.float32)
        for i, attr in enumerate(attr_names):
            boundary_path = os.path.join("../pretrained_models/stylegan2_ffhq_boundary",
                                         f"boundary_{attr}.npy")
            boundary[i] = np.load(boundary_path)
        direction2 = torch.from_numpy(boundary * 3.0)

    elif args.direction_model == 'IALS':
        attr_names = ['pose', 'eyeglasses', 'young', 'male', 'smiling']
        pretrain_root = "../pretrained_models/IALS_pretrain/attr_level_direction/interfacegan/ffhq"
        direction2 = torch.zeros((5,512), dtype=torch.float)
        for i, attr in enumerate(attr_names):
            model_path2 = os.path.join(pretrain_root, "%s.npy" % attr)
            if attr == "smile":
                direction2[i, :] = -torch.tensor(np.load(model_path2), dtype=torch.float)
            else:
                direction2[i, :] = torch.tensor(np.load(model_path2), dtype=torch.float)


    alpha_list = args.alpha_list



    EXPERIMENT_DATA_ARGS = {
        'ffhq_encode_pDSp': {
            "model_path": model_path,
            "image_path": args.sample_celeba_test
        },
        # "cars_encode": {
        #     "model_path": os.path.join(pretrained_models_path, "e4e_cars_encode.pt"),
        #     "image_path": "notebooks/images/car_img.jpg"
        # },
        # "horse_encode": {
        #     "model_path": os.path.join(pretrained_models_path, "e4e_horse_encode.pt"),
        #     "image_path": "notebooks/images/horse_img.jpg"
        # },
        # "church_encode": {
        #     "model_path": os.path.join(pretrained_models_path, "e4e_church_encode.pt"),
        #     "image_path": "notebooks/images/church_img.jpg"
        # }
    }


    data_paths = sorted(data_utils.make_dataset(args.sample_celeba_test))

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[args.experiment_type]
    if args.experiment_type == 'cars_encode':
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 192)
    else:
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 256)

    if args.experiment_type == 'ffhq_encode_pSCp':
        net = pSCp(opts)
    elif args.experiment_type == 'ffhq_encode_pDSp':
        net = pDSp(opts)
    else:
        raise Exception('experiment type is not defined!')

    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    if exp_type in ['SeFa', 'ganspace']:
        editor = latent_editor.LatentEditor(net.decoder, is_cars=False)
        # GANSpace
        # Here we provide the editings for the cars domain as displayed in the paper, as well as several examples for the facial domain,
        # taken from the official GANSpace repository.
        if exp_type == 'ganspace':
            ganspace_pca = torch.load('./editings/ganspace_pca/ffhq_pca.pt')
            directions = {
                'eye_openness': (54, 7, 8, 20),
                'trimmed_beard': (58, 7, 9, 20),
                'white_hair': (57, 7, 10, -24),
                'lipstick': (34, 10, 11, 20),
                'emotion_Happiness': (46, 4, 5, -20),
                'age': (20, 6, 7, 8),
                'pose_Yaw': (1, 0, 18, 0.5),
                'gender': (0, 0, 18, -1.5),
                'glass': (12, 0, 4, 10),
            }

    def display_alongside_source_image(result_image, source_image):
        res = np.concatenate([np.array(result_image),
                              np.array(source_image)], axis=1)
        return Image.fromarray(res)

    def run_on_batch(inputs, attributes, net, feat_2d=False):
        if feat_2d:
            images, latents = net.forward_D(inputs.to("cuda").float(), attributes.to("cuda"),
                randomize_noise=False, return_latents=True)
        else:
            images, latents = net.forward_Attr(inputs.to("cuda").float(), attributes.to("cuda"),
                randomize_noise=False, return_latents=True)

        if args.experiment_type == 'cars_encode':
            images = images[:, :, 32:224, :]
        return images, latents



    ## evaluation metrics
    loss_func_lpips = LPIPS(net_type='alex')
    loss_func_l2 = torch.nn.MSELoss()
    # face identity
    facenet = IR_101(input_size=112)
    facenet.load_state_dict(torch.load("../pretrained_models/CurricularFace_Backbone.pth"))
    facenet.cuda()
    facenet.eval()
    face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

    scores_dict = {}
    all_scores_lpips = []
    all_scores_l2 = []
    all_scores_psnr = []
    all_scores_ssim = []
    all_scores_id_edit = []
    all_scores_l2_edit = []
    all_scores_lpips_edit = []

    for image_path in data_paths:
        # image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
        file_name = image_path.split('/')[-1].split('.')[0]
        print(f"img: {image_path}, {file_name}")
        original_image = Image.open(image_path)
        original_image = original_image.convert("RGB")
        input_image = original_image
        input_image = input_image.resize(resize_dims)

        img_transforms = EXPERIMENT_ARGS['transform']
        transformed_image = img_transforms(input_image)
        transformed_image = transformed_image.unsqueeze(0)
        if exp_type in ['SeFa', 'ganspace']:
            semantic_ind = {
                'emotion_Happiness': (21, -1),
                'pose_Yaw': (4, 3),
                'gender': (2, 2),
                'age': (18, 3),
                'glass': (3, 3),
            }
            attributes_0 = torch.zeros(1, 512)
            _, latents_rec = run_on_batch(transformed_image, attributes_0, net)
            for att in ['pose_Yaw', 'glass', 'gender', 'emotion_Happiness', 'age']:
                attributes_0 = torch.zeros(1, 512)
                for alpha in alpha_list:
                    # _, latents_rec = run_on_batch(transformed_image.unsqueeze(0), attributes_0, net)
                    if exp_type == 'SeFa':
                        images = editor.apply_sefa(latents_rec, indices='all', semantics=semantic_ind[att][0],
                                                   start_distance=alpha*semantic_ind[att][1], end_distance=15.0, step=1)
                    elif exp_type == 'ganspace':
                        edit_direction = directions[att]
                        images = editor.apply_ganspace(latents_rec, ganspace_pca, [edit_direction], alpha)
                    ## evaluate
                    if alpha == 0:
                        ## construction
                        loss_lpips = float(loss_func_lpips(images.cuda(), transformed_image.cuda()))
                        loss_l2 = float(loss_func_l2(images.cuda(), transformed_image.cuda()))
                        psnr, ssim = calc_psnr(images.cuda(), transformed_image.cuda(), scale=4, rgb_range=1, dataset=None)
                        all_scores_lpips.append(loss_lpips)
                        all_scores_l2.append(loss_l2)
                        all_scores_psnr.append(psnr)
                        all_scores_ssim.append(ssim)
                    else:
                        ## edit
                        input_id = facenet(face_pool(images[:, :, 35:223, 32:220].cuda()))[0]
                        result_id = facenet(face_pool(transformed_image[:, :, 35:223, 32:220].cuda()))[0]
                        all_scores_id_edit.append(float(input_id.dot(result_id)))
                        loss_l2_edit = float(loss_func_l2(images.cuda(), transformed_image.cuda()))
                        loss_lpips_edit = float(loss_func_lpips(images.cuda(), transformed_image.cuda()))
                        all_scores_l2_edit.append(loss_l2_edit)
                        all_scores_lpips_edit.append(loss_lpips_edit)
                    dis_img = tensor2im(images[0])
                    output_path = os.path.join(exp_path, file_name + f'_{alpha}_{att}.jpg') #00000_1_age_edit_single
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    dis_img.save(output_path)

        elif exp_type == "edit_single":
            for attributes_idx in args.edit_attributes:
                attributes = torch.zeros(1, opts.num_attributes)
                result_images = []
                for alpha in alpha_list:
                    # ## multi
                    # attributes_idxs = [1, 4]
                    # for attributes_idx in attributes_idxs:
                    #     attributes[0, attributes_idx] = alpha
                    # print(f"alpha: {alpha}")
                    ## single
                    attributes[0, attributes_idx] = alpha
                    with torch.no_grad():
                        tic = time.time()
                        ## ori
                        attributes2 = torch.matmul(attributes.to("cuda"), direction2.to("cuda"))#.unsqueeze(0)

                        images, latents = run_on_batch(transformed_image, attributes2, net, feat_2d=True)
                        toc = time.time()

                        ## evaluate
                        if args.evaluate and attributes_idx != 0:
                            if alpha == 0:
                                ## construction
                                loss_lpips = float(loss_func_lpips(images.cuda(), transformed_image.cuda()))
                                loss_l2 = float(loss_func_l2(images.cuda(), transformed_image.cuda()))
                                psnr, ssim = calc_psnr(images.cuda(), transformed_image.cuda(), scale=4, rgb_range=1, dataset=None)
                                all_scores_lpips.append(loss_lpips)
                                all_scores_l2.append(loss_l2)
                                all_scores_psnr.append(psnr)
                                all_scores_ssim.append(ssim)
                            elif alpha in [-1, 1]:
                                ## edit
                                input_id = facenet(face_pool(images[:, :, 35:223, 32:220].cuda()))[0]
                                result_id = facenet(face_pool(transformed_image[:, :, 35:223, 32:220].cuda()))[0]
                                all_scores_id_edit.append(float(input_id.dot(result_id)))
                                loss_l2_edit = float(loss_func_l2(images.cuda(), transformed_image.cuda()))
                                loss_lpips_edit = float(loss_func_lpips(images.cuda(), transformed_image.cuda()))
                                all_scores_l2_edit.append(loss_l2_edit)
                                all_scores_lpips_edit.append(loss_lpips_edit)

                        result_image, latent = images[0], latents[0]

                    # Display inversion:
                    if args.save_gt:
                        output_path_gt = os.path.join(exp_path_gt, file_name + f'_{alpha}_{select_attributes[attributes_idx]}.jpg')
                        os.makedirs(os.path.dirname(output_path_gt), exist_ok=True)
                        input_image.save(output_path_gt)
                    dis_img = tensor2im(result_image)
                    output_path = os.path.join(exp_path, file_name + f'_{alpha}_{select_attributes[attributes_idx]}.jpg')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    dis_img.save(output_path)
                    if args.save_rec and alpha == 0:
                        output_path_rec = os.path.join(args.exp_path_rec, file_name + f'_{alpha}_{select_attributes[attributes_idx]}.jpg')
                        os.makedirs(os.path.dirname(output_path_rec), exist_ok=True)
                        dis_img.save(output_path_rec)


        elif exp_type == "edit_multi":
            attributes = torch.zeros(1, opts.num_attributes)
            result_images = []
            for alpha in alpha_list:
                for attributes_idx in attributes_idxs:
                    attributes[0, attributes_idx] = alpha
                with torch.no_grad():
                    tic = time.time()
                    images, latents = run_on_batch(transformed_image.unsqueeze(0), attributes, net)
                    toc = time.time()

                    result_image, latent = images[0], latents[0]
                    if result_images:
                        result_images = display_alongside_source_image(result_images, tensor2im(result_image))
                    else:
                        result_images = tensor2im(result_image)
            # Display inversion:
            dis_img = display_alongside_source_image(result_images, input_image)
            output_path = os.path.join(exp_path, file_name + f'_{attributes_save_names}_{exp_type}.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dis_img.save(output_path)
        else:
            ## load attribute GT
            idx = attribute_data['img'].index(f"{file_name}.txt")
            attributes = []
            for attr in select_attributes:
                attributes.append(attribute_data[attr][idx])
            attributes = torch.Tensor(attributes)
            attributes = attributes.unsqueeze(0)

            result_images = []
            with torch.no_grad():
                tic = time.time()
                images, latents = run_on_batch(transformed_image.unsqueeze(0), attributes, net)
                toc = time.time()
                # print('Inference took {:.4f} seconds.'.format(toc - tic))

                result_image, latent = images[0], latents[0]
                if result_images:
                    result_images = display_alongside_source_image(result_images, tensor2im(result_image))
                else:
                    result_images = tensor2im(result_image)
            # Display inversion:
            dis_img = display_alongside_source_image(result_images, input_image)
            output_path = os.path.join(exp_path, file_name + f'_{exp_type}.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dis_img.save(output_path)


        # if int(file_name) > args.num_img:
        #     break

    ## save results
    scores_dict['lpips'] = all_scores_lpips
    scores_dict['l2'] = all_scores_l2
    scores_dict['psnr'] = all_scores_psnr
    scores_dict['ssim'] = all_scores_ssim
    scores_dict['id_edit'] = all_scores_id_edit
    scores_dict['l2_edit'] = all_scores_l2_edit
    scores_dict['lpips_edit'] = all_scores_lpips_edit
    result_str = "Results: \n"
    for key, value in scores_dict.items():
        result_str += 'Average {} is {:.4f}+-{:.4f}\n'.format(key, np.mean(value), np.std(value))

    print(result_str)
    save_results.write(result_str)
    save_results.close()
    with open(os.path.join(exp_path, "results.json"), 'w') as outfile:
        json.dump(scores_dict, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--gpus', type=str, default='0', help='ffhq_encode_pDSp, car_encode_')

    parser.add_argument('--experiment_type', type=str, default="ffhq_encode_pDSp", help='ffhq_encode_pDSp, car_encode_')
    parser.add_argument('--sample_celeba_test', type=str, default="../../Dataset/FaceRepresentation/celeba_hq/val/all/",
                        help='ffhq/resized256/ or celeba_hq/val/all/')
    parser.add_argument('--pretrained_models_path', type=str,
                        default="../pretrained_models/pretrained/iteration_60000.pt", help='save output results')
    parser.add_argument('--enc_model', type=str, default="mine", help='mine, psp, e4e')
    parser.add_argument('--direction_model', type=str, default="mine", help='mine, InterfaceGAN, IALS, SeFa, ganspace')
    parser.add_argument('--exp_path', type=str, default="../experiments/test",
                        help='output file')
    parser.add_argument('--alpha_list', type=list, default=[0, 1, 2], help='alpha_list: [0, 1, 2]')
    parser.add_argument('--edit_attributes', type=list, default=[1, 2, 3, 4],
                        help="'pose_Yaw', 'glass', 'age', 'gender', 'emotion_Happiness', -1 for all, 0, 1, 2, 3, 4")

    parser.add_argument('--num_img', type=int, default=10000, help='num of images')
    parser.add_argument('--save_gt', action='store_true', default=False, help='save gt images')
    parser.add_argument('--save_rec', action='store_true', default=False, help='save reconstruct images')
    parser.add_argument('--evaluate', action='store_true', default=False, help='compute evaluate metric')

    args = parser.parse_args()
    if args.edit_attributes == [-1]:
         args.edit_attributes= range(opts.num_attributes)

    args.exp_path_gt = os.path.join(args.exp_path, "Ori_img")
    args.exp_path_rec = os.path.join(args.exp_path, "Ori_rec")

    # run()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    run_seperate_enc_dir(args)
