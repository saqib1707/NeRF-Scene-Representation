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
np.random.seed(0)
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


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
#     parser.add_argument("--basedir", type=str, default=os.path.join(os.getcwd(), "drive/MyDrive/CSE291/assignments/assignment3/"), 
#                         help='project base directory')
#     parser.add_argument("--logdir", type=str, default=os.path.join(os.getcwd(), "drive/MyDrive/CSE291/assignments/assignment3/logdir/"), 
#                         help='where to store ckpts and logs')
#     parser.add_argument("--datadir", type=str, default=os.path.join(os.getcwd(), "drive/MyDrive/CSE291/assignments/assignment3/data/bottles"), 
#                         help='input data directory')

    parser.add_argument("--basedir", type=str, default=os.path.join(os.getcwd(), "../"), 
                        help='project base directory')
    parser.add_argument("--logdir", type=str, default=os.path.join(os.getcwd(), "../logdir/"), 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default=os.path.join(os.getcwd(), "../data/bottles"), 
                        help='input data directory')

    # network architecture options
    parser.add_argument("--netdepth", type=int, default=4, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=4, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128, 
                        help='channels per layer in fine network')

    # ray batch options
    parser.add_argument("--ray_batch_size", type=int, default=32*32, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=1024, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    
    # training options
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay_steps", type=int, default=250, 
                        help='exponential learning rate decay steps (in 1000 steps)')
    parser.add_argument("--decay_rate", type=float, default=0.1, 
                        help='exponential learning rate decay')
    
    # weights/checkpoint options
    parser.add_argument("--no_reload_ckpt", action='store_true', 
                        help='do not reload weights from saved ckpt file (if exists)')
    parser.add_argument("--ckpt_coarsenet_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--num_samples_per_ray", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--num_fine_samples_per_ray", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    
    # positional encoding
    parser.add_argument("--num_freqs_xyz", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--num_freqs_viewdir", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')

    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--num_train_itr", type=int, default=10000,
                        help='number of steps to train')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--resize_factor", type=int, default=1, 
                        help='resize image by factor')
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=1, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train(train_data_dict, test_data_dict, cam_int_mat, num_itr=100000+1):
    train_images = train_data_dict['images']
    train_poses = train_data_dict['poses']
#     test_poses = train_data_dict['poses'][0:5]
    test_poses = test_data_dict['poses']
    test_reqd_poses = np.array([test_poses[0], test_poses[16], test_poses[55], test_poses[93], test_poses[160]])
    
    print("Loaded dataset:", train_images.shape, train_poses.shape, test_poses.shape)
    
    print('DEFINING BOUNDS')
    near = 0.
    far = 5.0
    
    if args.white_bkgd:
        train_images = train_images[...,:3]*train_images[...,-1:] + (1.0 - train_images[...,-1:])
    else:
        train_images = train_images[...,:3]

    img_height, img_width, focal = train_images[0].shape[0], train_images[1].shape[1], cam_int_mat[0,0]
    
    # create the log directory and store the hyperparameters in config file
    os.makedirs(os.path.join(args.logdir, args.expname), exist_ok=True)
    args_file = os.path.join(args.logdir, args.expname, "args.txt")
    
    with open(args_file, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
            
    if args.config is not None:
        config_file = os.path.join(args.logdir, args.expname, 'config.txt')
        with open(config_file, 'w') as file:
            file.write(open(args.config, 'r').read())

    # create NeRF model
    render_kwargs_train, render_kwargs_test, start_itr, grad_vars, optimizer = create_NeRF(args)
    render_kwargs_train.update({'near' : near, 'far' : far,})
    render_kwargs_test.update({'near' : near, 'far' : far,})
    
    global_step = start_itr
    # move test data poses to GPU
    test_poses = torch.tensor(test_poses).to(device)
    
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("Render Only Test Poses")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                test_images = train_images[0]
            else:
                # Default is smoother render_poses path
                test_images = None

            testsavedir = os.path.join(args.logdir, args.expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 
                                                                                                'path', start_itr))
            os.makedirs(testsavedir, exist_ok=True)
            print("Test poses shape:", test_poses.shape)

            rgbs, _ = render_path(test_poses, img_height, img_width, focal, K, args.chunk, render_kwargs_test, 
                                  gt_imgs=test_images, savedir=testsavedir, render_factor=args.render_factor)
            
            print("Rendering Complete !!!", testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return
    
    # prepare ray batch tensor if batching random rays
    use_batching = not args.no_batching
    
    if use_batching:
        # for random ray batching
        print("Using ray batching --> Obtaining rays !!!")
        rays = np.stack([get_rays_np(img_height, img_width, cam_int_mat, p) for p in train_poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print("Rays obtained --> Concatenate", rays.shape)
        rays_rgb = np.concatenate([rays, train_images[:,None]], 1)     # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])      # [N, H, W, ro+rd+rgb, 3]
#         rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3])     # [N*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0
        
        # Move training data to GPU
        train_images = torch.tensor(train_images).to(device)
        rays_rgb = torch.tensor(rays_rgb).to(device)
    
    train_poses = torch.tensor(train_poses).to(device)
    
    start_itr = start_itr + 1
    print("Start Training !!!")
    
    # default `log_dir` is "runs"
    writer = SummaryWriter(os.path.join(args.logdir, args.expname, 'runs/'))
    
    for i in tqdm(range(start_itr, num_itr)):
        start_time = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + args.ray_batch_size] # [ray_batch_size, 2+1, 3]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += args.ray_batch_size
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            # Randomly choose one training image and compute rays
            img_idx = np.random.choice(range(0, train_images.shape[0]))
            target = train_images[img_idx]
            target = torch.tensor(target).to(device)
            pose = train_poses[img_idx, :3,:4]

            if args.ray_batch_size is not None:
                rays_o, rays_d = get_rays(img_height, img_width, cam_int_mat, pose)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(img_height//2 * args.precrop_frac)
                    dW = int(img_width//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(img_height//2 - dH, img_height//2 + dH - 1, 2*dH), 
                            torch.linspace(img_width//2 - dW, img_width//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start_itr:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, img_height-1, img_height), 
                                                        torch.linspace(0, img_width-1, img_width)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[args.ray_batch_size], replace=False)  # (ray_batch_size,)
                select_coords = coords[select_inds].long()  # (ray_batch_size, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (ray_batch_size, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (ray_batch_size, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (ray_batch_size, 3)
        
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(img_height, img_width, cam_int_mat, chunk=args.chunk, rays=batch_rays, 
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
#         trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
            
        loss.backward()
        optimizer.step()
        global_step += 1

        ###   NOTE: IMPORTANT! update learning rate   ###
#         decay_steps = args.lrate_decay_steps * 1000
#         new_lrate = args.lrate * (args.decay_rate ** (global_step / decay_steps))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = new_lrate
        
        dt = round(time.time() - start_time, 4)
        #####           end            #####

        # ...log the running loss
        writer.add_scalar('training loss', loss.item(), global_step)
        writer.add_scalar('psnr', psnr.item(), global_step)
            
        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(args.logdir, args.expname, '{:06d}.tar'.format(i))
            
            if args.num_fine_samples_per_ray > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            
            print('Saved checkpoints at:', path)
        
        if i % args.i_video == 0 and i > 0:
            print("Synthesizing views for test poses...")
            
            testsavedir = os.path.join(args.logdir, args.expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('Test poses shape:', test_poses.shape)

            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(test_poses, img_height, img_width, focal, cam_int_mat, args.chunk, render_kwargs_test,
                                         gt_imgs=None, savedir=testsavedir)
            
            print('Done, saving', rgbs.shape, disps.shape)
            
            moviebase = os.path.join(args.logdir, args.expname, '{}_spiral_{:06d}_'.format(args.expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        
#         if i % args.i_testset == 0 and i > 0:
            
#             with torch.no_grad():
#                 render_path(test_poses.to(device), img_height, img_width, focal, cam_int_mat, args.chunk, render_kwargs_test, 
#                             gt_imgs=None, savedir=testsavedir)
#             print('Saved test set')
        
        
        if i % args.i_print==0:
            print(f"[TRAIN] Iter: {i} / {num_itr}, Loss: {loss.item()}, PSNR: {psnr.item()}, Time: {dt} s")
       

    
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args(["--expname=exp005", 
                              "--use_viewdirs", 
                              "--i_weights=10000", 
                              "--i_video=50000", 
                              "--i_print=10000", 
                              "--i_testset=50000", 
                              "--resize_factor=4", 
                              "--num_train_itr=200001"])
    
    print("Loading dataset...")
    train_data_dict, val_data_dict, test_data_dict, cam_int_mat, bbox_mat = dataset.load_data_from_files(store_data=False,
                                                                                          resize_factor=args.resize_factor)
    print("Dataset loaded !!!")

    print("Training data:", train_data_dict['images'].shape, train_data_dict['poses'].shape)
    print("Validation data:", val_data_dict['images'].shape, val_data_dict['poses'].shape)
    print("Test data:", test_data_dict['poses'].shape)
    print("Cam Intrinsic mat and bbox:", cam_int_mat.shape, bbox_mat.shape)

    train(train_data_dict, test_data_dict, cam_int_mat, num_itr=args.num_train_itr)
    