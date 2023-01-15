import os
print("Current working directory:", os.getcwd())
import sys
# sys.path.append('/content/drive/MyDrive/CSE291/assignments/assignment3/')
# sys.path.append('/content/drive/MyDrive/CSE291/assignments/assignment3/code/')

import time
import math
import argparse
from PIL import Image
import imageio
import pickle
from tqdm.notebook import tqdm
import importlib

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataset
import nerf_utils
from nerf_utils import *


# define device type - cuda:0 or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Additional Info when using cuda
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
else:
    print("Device:", device)


def get_configs(base_dir=os.path.join(os.getcwd(), "../")):
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, help='experiment name')
    parser.add_argument("--base_dir", type=str, default=base_dir, help='project base directory')
    parser.add_argument("--log_dir", type=str, default=os.path.join(base_dir, "logdir/"), help='To store checkpoints and results')
    parser.add_argument("--data_dir", type=str, default=os.path.join(base_dir, "data/bottles"), help='input data directory')

    # network architecture options
    parser.add_argument("--num_layers", type=int, default=4, help='number of layers in coarse network')
    parser.add_argument("--num_channels", type=int, default=128, help='number of channels/layer in coarse network')
    parser.add_argument("--num_layers_fine", type=int, default=4, help='number of layers in fine network')
    parser.add_argument("--num_channels_fine", type=int, default=128, help='number of channels/layer in fine network')

    # ray batch options
    parser.add_argument("--ray_batch_size", type=int, default=32*32, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=32*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=32*32, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--use_batching", action='store_false', help='only take random rays from 1 image at a time')
    
    # training options
    parser.add_argument("--num_train_itr", type=int, default=10000, help='number of steps to train')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_steps", type=int, default=250, help='exponential learning rate decay steps (in 1000 steps)')
    parser.add_argument("--decay_rate", type=float, default=0.1, help='exponential learning rate decay')
    
    # weights/checkpoint options
    parser.add_argument("--no_reload_ckpt", action='store_true', help='do not reload weights from saved ckpt file (if exists)')
    parser.add_argument("--ckpt_coarsenet_path", type=str, default=None, help='saved weights path for coarse network')

    # rendering options
    parser.add_argument("--num_samples_per_ray", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--num_fine_samples_per_ray", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", type=bool, default=True, help='use full 5D input instead of 3D')
    
    # positional encoding
    parser.add_argument("--num_freqs_xyz", type=int, default=5, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--num_freqs_viewdir", type=int, default=2, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--use_embedding", type=bool, default=True, help='set to True for default positional encoding, False for none')

    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # dataset options
    parser.add_argument("--resize_factor", type=int, default=1, help='resize image by factor')
    parser.add_argument("--is_bkgd_white", type=bool, default=False, help='set to true when rendering images on a white bkgd')

    parser.add_argument("--sample_disp_linear", bool=False, help='sampling linearly in disparity rather than depth')

    # logging/saving options
    parser.add_argument("--log_print_idx",   type=int, default=1000, help='frequency of console printout and metric loggin')
    parser.add_argument("--save_ckpts_idx", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--val_idx",   type=int, default=50000, help='frequency of render_poses video saving')

    return parser



def train(train_data_dict, val_data_dict, test_data_dict, cam_int_mat, num_itr=100000+1, args=None):
    train_images = train_data_dict['images']    # [N, H, W, 3]
    train_poses = train_data_dict['poses']      # [N, 4, 4]
    
    val_images = val_data_dict['images']
    val_poses = val_data_dict['poses']

    test_poses = test_data_dict['poses']        # [num_test_poses, 4, 4]
    test_poses = np.array([test_poses[0], test_poses[16], test_poses[55], test_poses[93], test_poses[160]])
    
#     print("Loaded dataset:", train_images.shape, train_poses.shape, test_poses.shape)
    
    near_dist = 0.
    far_dist = 5.0
    
    if args.is_bkgd_white:
        train_images = train_images[...,:3] * train_images[...,-1:] + (1.0 - train_images[...,-1:])
        val_images = val_images[...,:3] * val_images[...,-1:] + (1.0 - val_images[...,-1:])
    else:
        train_images = train_images[...,:3]
        val_images = val_images[...,:3]

    img_height = train_images[0].shape[0]
    img_width = train_images[1].shape[1]
    focal = cam_int_mat[0,0]
    
    # create the log directory and store the hyperparameters in config file
    os.makedirs(os.path.join(args.log_dir, args.exp_name), exist_ok=True)
    args_file = os.path.join(args.log_dir, args.exp_name, "args.txt")
    
    with open(args_file, 'w') as file:
        for argument in sorted(vars(args)):
            attr = getattr(args, argument)
            file.write('{} = {}\n'.format(argument, attr))


    # create NeRF model
    render_train_options, render_test_options, start_itr, grad_vars, optimizer = create_NeRF(args)
    render_train_options.update({'near' : near, 'far' : far,})
    render_test_options.update({'near' : near, 'far' : far,})
    
    global_step = start_itr
    
    # move test data poses to GPU
    test_poses = torch.tensor(test_poses).to(device)
    val_poses = torch.tensor(val_poses).to(device)
#     val_images = torch.tensor(val_images).to(device)

    
    # prepare ray batch tensor if batching random rays
    if args.use_batching:
        # for random ray batching
        print("Using ray batching --> Obtaining rays !!!")
        rays = np.stack([compute_rays(img_height, img_width, cam_int_mat, p) for p in train_poses[:,:3,:4]], 0) # [N, ro+rd=2, H, W, 3]
        print("Rays obtained --> Concatenate", rays.shape)
#         print(train_images[:,None].shape)     # [N, 1, H, W, 3]
        
        rays_rgb = np.concatenate([rays, train_images[:,None]], 1)     # [N, ro+rd+rgb=3, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])      # [N, H, W, ro+rd+rgb=3, 3]
#         rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3])     # [N*H*W, ro+rd+rgb=3, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0
        
        # Move training data to GPU
        train_images = torch.tensor(train_images).to(device)   # [N, H, W, 3]
        rays_rgb = torch.tensor(rays_rgb).to(device)           # [N*H*W, ro+rd+rgb=3, 3]
    
    train_poses = torch.tensor(train_poses).to(device)
    
    start_itr = start_itr + 1
    print("Start Training !!!")
    
    # default `log_dir` is "runs"
    writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name, 'runs/'))
    
    for itr in tqdm(range(start_itr, num_itr)):
        start_time = time.time()

        # Sample random ray batch
        if args.use_batching:
            batch = rays_rgb[i_batch:i_batch + args.ray_batch_size] # [ray_batch_size, ro+rd+rgb=3, 3]
            batch = torch.transpose(batch, 0, 1)                    # [ro+rd+rgb=3, ray_batch_size, 3]
            batch_rays, target_s = batch[:2], batch[2]              # [ro+rd=2, ray_batch_size, 3], [rgb=1, ray_batch_size, 3]

            i_batch += args.ray_batch_size
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        
        # Training optimization
        img_synthesized, disp_map_synthesized, acc_map_synthesized, _ = render(img_height, img_width, cam_int_mat, chunk=args.chunk, rays=batch_rays, **render_train_options)

        # reset the optimizer's gradients
        optimizer.zero_grad()
        loss_val = compute_mse(img_synthesized, target_s)
        train_psnr = compute_psnr_from_mse(img_loss)
        
        # if 'rgb0' in extras:
        #     img_loss0 = compute_mse(extras['rgb0'], target_s)
        #     loss_val += img_loss0
            
        loss_val.backward()
        optimizer.step()
        global_step += 1

        # update learning rate
        decay_steps = args.lrate_decay_steps * 1000
        new_lrate = args.lrate * (args.decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        time_elapsed = round(time.time() - start_time, 4)

        # log the training loss and training psnr
        writer.add_scalar('training loss', loss_val.item(), global_step)
        writer.add_scalar('train psnr', train_psnr.item(), global_step)

        # Save weights and checkpoints
        if itr % args.save_ckpts_idx == 0:
            path = os.path.join(args.log_dir, args.exp_name, '{:06d}.tar'.format(i))
            
            if args.num_fine_samples_per_ray > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_train_options['network_fn'].state_dict(),
                    'network_fine_state_dict': render_train_options['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_train_options['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            
            print('Saved checkpoints at:', path)
        
        
        if itr % args.val_idx == 0 and i > 0:
            # evaluation on validation images
            print("Synthesizing novel views for validation poses...")
            
            valsavedir = os.path.join(args.log_dir, args.exp_name, 'valset_{:06d}'.format(i))
            os.makedirs(valsavedir, exist_ok=True)
            print('Val poses shape:', val_poses.shape)
            
            # Turn on Validation mode
            with torch.no_grad():
                syn_rgb_imgs, syn_disp_maps = synthesize_imgs(val_poses, img_height, img_width, cam_int_mat, args.chunk, render_test_options, gt_imgs=None, savedir=valsavedir)
            
            # compute validation PSNR
            val_psnr = 0.
            val_images = val_images.astype(np.float32)
            for val_idx in range(val_images.shape[0]):
                val_psnr += -10. * np.log10(np.mean((syn_rgb_imgs[val_idx] - val_images[val_idx]) ** 2))
#                 val_psnr += compute_psnr_from_mse(compute_mse(syn_rgb_imgs[val_idx], val_images[val_idx]))
            
            val_psnr /= val_images.shape[0]
            writer.add_scalar('Val psnr', val_psnr, global_step)
            
            print('Done saving synthesized images:', val_psnr)
            
            moviebase = os.path.join(args.log_dir, args.exp_name, '{}_valset_{:06d}_'.format(args.exp_name, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', convert_np_to_img(syn_rgb_imgs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', convert_np_to_img(syn_disp_maps / np.max(syn_disp_maps)), fps=30, quality=8)
            

#         if i % args.val_idx == 0 and i > 0:
#             print("Synthesizing views for test poses...")
            
#             testsavedir = os.path.join(args.log_dir, args.exp_name, 'testset_{:06d}'.format(i))
#             os.makedirs(testsavedir, exist_ok=True)
#             print('Test poses shape:', test_poses.shape)

#             # Turn on testing mode
#             with torch.no_grad():
#                 test_img_height = img_height * args.resize_factor
#                 test_img_width = img_width * args.resize_factor
#                 test_cam_int_mat = cam_int_mat * args.resize_factor
                
#                 rgbs, disps = synthesize_imgs(test_poses, test_img_height, test_img_width, test_cam_int_mat, args.chunk, 
#                                           render_test_options, gt_imgs=None, savedir=testsavedir)
            
#             print('Done saving synthesized images:', rgbs.shape, disps.shape)
            
#             moviebase = os.path.join(args.log_dir, args.exp_name, '{}_testset_{:06d}_'.format(args.exp_name, i))
#             imageio.mimwrite(moviebase + 'rgb.mp4', convert_np_to_img(rgbs), fps=30, quality=8)
#             imageio.mimwrite(moviebase + 'disp.mp4', convert_np_to_img(disps / np.max(disps)), fps=30, quality=8)
        
        
        if itr % args.log_print_idx==0:
            print("[Train] Itr:", itr, "/", num_itr, ", Loss: ", loss_val.item(), ", Train PSNR: ", train_psnr.item(), ", Time: ", time_elapsed, "s")


def synthesize_imgs(test_poses, img_height, img_width, cam_int_mat, chunk, render_options, gt_imgs=None, savedir=None):
    print("Synthesize images from given poses")

    synthesized_imgs = []
    synthesized_disp_maps = []

    for i, cam_pose in enumerate(tqdm(test_poses)):
        rgb, disp, acc, _ = render(img_height, img_width, cam_int_mat, chunk=chunk, cam_pose=cam_pose[:3,:4], **render_options)
        
        synthesized_imgs.append(rgb.cpu().numpy())
        synthesized_disp_maps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8 = convert_np_to_img(synthesized_imgs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    synthesized_imgs = np.stack(synthesized_imgs, 0)
    synthesized_disp_maps = np.stack(synthesized_disp_maps, 0)

    return synthesized_imgs, synthesized_disp_maps


    
if __name__ == "__main__":
    parser = get_configs()
    args = parser.parse_args(["--exp_name=finalexp002_whitebkgd", 
                              "--save_ckpts_idx=10000", 
                              "--val_idx=20000",
                              "--log_print_idx=1000", 
                              "--resize_factor=4", 
                              "--num_train_itr=200001", 
                              "--is_bkgd_white=True"])
    
    print("Loading dataset...")
    train_data_dict, val_data_dict, test_data_dict, cam_int_mat, bbox_mat = dataset.load_data_from_files(store_data=False,
                                                                                          resize_factor=args.resize_factor)
    print("Dataset loaded !!!")

    print("Training data:", train_data_dict['images'].shape, train_data_dict['poses'].shape)
    print("Validation data:", val_data_dict['images'].shape, val_data_dict['poses'].shape)
    print("Test data:", test_data_dict['poses'].shape)
    print("Cam Intrinsic mat and bbox:", cam_int_mat.shape, bbox_mat.shape)

    train(train_data_dict, val_data_dict, test_data_dict, cam_int_mat, num_itr=args.num_train_itr, args=args)
    