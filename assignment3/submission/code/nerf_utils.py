import os
import sys
import time
from PIL import Image
import imageio
from tqdm.notebook import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# define device type - cuda:0 or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_mse(x, y):
    return torch.mean((x - y) ** 2)


def compute_psnr_from_mse(x):
    return -10.0 * torch.log10(x)


def convert_np_to_img(x):
    return (255.0 * np.clip(x, 0, 1.0)).astype(np.uint8)


def run_network(inputs, viewdirs, fn, posenc_xyz_fn, posenc_dir_fn, netchunk=1024*64):
    """
    Arguments:
        inputs: [num_rays, num_samples_per_ray, 3]
        viewdirs: [num_rays, 3]
        posenc_xyz_fn: positional encoding function for xyz coordinates
        posenc_dir_fn: positional encoding function for direction (d1,d2,d3) coordinates
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [num_rays * num_samples_per_ray, 3]
    embedded_xyz = posenc_xyz_fn(inputs_flat)                         # [num_rays * num_samples_per_ray, 3 * num_freqs_xyz * 2 + 3]

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)   # [num_rays, num_samples_per_ray, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # [num_rays * num_samples_per_ray, 3]
        embedded_dirs = posenc_dir_fn(input_dirs_flat)         # [num_rays * num_samples_per_ray, 3 * num_freqs_dirs * 2 + 3]
        embedded = torch.cat([embedded_xyz, embedded_dirs], dim=-1)  # [num_rays * num_samples_per_ray, (3 * num_freqs_xyz * 2 + 3) + (3 * num_freqs_dirs * 2 + 3)]
        outputs_flat = embedded
    else:
        outputs_flat = embedded_xyz

    outputs_flat = torch.cat([fn(outputs_flat[i:i + netchunk]) for i in range(0, outputs_flat.shape[0], netchunk)], dim=0)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs


def convert_rays_to_batch(rays_flat, chunk=1024*32, **kwargs):
    """
    rays_flat: [ray_batch_size, 3+3+2+3 = 11 (ro+rd+near_dist+far_dist+normalized_viewdir)]
    """
    all_ret = {}
    
    for i in range(0, rays_flat.shape[0], chunk):     # (start, stop, step)
        ret = stratified_sampling(rays_flat[i:i+chunk], **kwargs)
        
        for key in ret:
            if key not in all_ret:
                all_ret[key] = []
            all_ret[key].append(ret[key])

    all_ret = {key : torch.cat(all_ret[key], dim=0) for key in all_ret}
    
    return all_ret


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, num_freqs, log_space=True, include_input=False):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.log_space = log_space
        self.include_input = include_input
        self.posenc_xyz_fn = [lambda x: x]
        
        if self.include_input:
            self.out_dim = self.input_dim * (2 * self.num_freqs + 1)
        else:
            self.out_dim = self.input_dim * (2 * self.num_freqs)

        if self.log_space:
            self.freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2 ** (self.num_freqs - 1), self.num_freqs)
            
        # Alternate sin and cos
        for freq in self.freq_bands:
            self.posenc_xyz_fn.append(lambda x, freq=freq: torch.sin(x * freq))
            self.posenc_xyz_fn.append(lambda x, freq=freq: torch.cos(x * freq))
    
    def forward(self, x):
        """
        apply positional encoding to input x
        """
#         self.posenc_xyz_fn = []
#         if self.include_input:
#             self.posenc_xyz_fn.append(x)
        
#         for freq in self.freq_bands:
#             self.posenc_xyz_fn.append(torch.sin(x * freq))
#             self.posenc_xyz_fn.append(torch.cos(x * freq))
        
        return torch.cat([fn(x) for fn in self.posenc_xyz_fn], dim=-1)


class NeRF(nn.Module):
    def __init__(self, num_layers=8, filter_dim=256, input_dim=3, output_dim=4, skip_layers=[3], viewdir_dim=3, 
                 use_viewdirs=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.filter_dim = filter_dim
        self.skip_layers = skip_layers
        self.viewdir_dim = viewdir_dim
        self.use_viewdirs = use_viewdirs
        
        self.relu_layer = nn.ReLU()
        
        # define network
        layers = []
        
        for i in range(self.num_layers-1):
            if i in self.skip_layers:
                layers.append(nn.Linear(filter_dim + input_dim, filter_dim))
            else:
                layers.append(nn.Linear(filter_dim, filter_dim))

        self.pts_layers = nn.ModuleList([nn.Linear(self.input_dim, filter_dim)] + layers)
        
        if self.use_viewdirs:
            self.sigma_layer = nn.Linear(filter_dim, 1)
            self.feature_layer = nn.Linear(filter_dim, filter_dim)
            self.viewdir_layers = nn.ModuleList([nn.Linear(filter_dim + self.viewdir_dim, filter_dim // 2)])
            self.rgb_layer = nn.Linear(filter_dim // 2, 3)
        else:
            self.output_layer = nn.Linear(filter_dim, output_dim)
        

    def forward(self, x):
        input_pts, viewdir = torch.split(x, [self.input_dim, self.viewdir_dim], dim=-1)
        
        y = torch.clone(input_pts)
        
        for i, layer in enumerate(self.pts_layers):
            y = self.relu_layer(layer(y))
            
            if i in self.skip_layers:
                y = torch.cat([input_pts, y], dim=-1)

        if self.use_viewdirs:
            sigma = self.relu_layer(self.sigma_layer(y))
            
            feature = self.feature_layer(y)
            y = torch.cat([feature, viewdir], dim=-1)
            
            for j, layer in enumerate(self.viewdir_layers):
                y = self.relu_layer(layer(y))  
            
            rgb = self.rgb_layer(y)
            output = torch.cat([rgb, sigma], dim=-1)
        else:
            output = self.output_layer(y)
        
        return output


def create_NeRF(args):
    
    # get positional encoding for the position (x,y,z) coordinates
    if args.use_embedding == False:
        posenc_xyz_fn, input_dim = nn.Identity(), 3
    else:
        embedder_obj = PositionalEncoder(input_dim=3, num_freqs=args.num_freqs_xyz, log_space=True, include_input=True)
        posenc_xyz_fn = lambda x, eo=embedder_obj : eo(x)
        input_dim = embedder_obj.out_dim
    
    viewdir_dim = 0
    posenc_dir_fn = None
    
    # get positional encoding for the view direction (d1, d2, d3 / theta, phi) coordinates
    if args.use_viewdirs:
        if args.use_embedding == False:
            posenc_dir_fn, viewdir_dim = nn.Identity(), 3
        else:
            embedder_obj = PositionalEncoder(input_dim=3, num_freqs=args.num_freqs_viewdir, log_space=True, include_input=True)
            posenc_dir_fn = lambda x, eo=embedder_obj : eo(x)
            viewdir_dim = embedder_obj.out_dim
    
    output_dim = 5 if args.num_fine_samples_per_ray > 0 else 4
    skip_layers = [args.num_layers//2 - 1]
    
    model_coarse = NeRF(num_layers=args.num_layers, 
                     filter_dim=args.num_channels, 
                     input_dim=input_dim, 
                     output_dim=output_dim, 
                     skip_layers=skip_layers, 
                     viewdir_dim=viewdir_dim, 
                     use_viewdirs=args.use_viewdirs).to(device)
    
    grad_vars = list(model_coarse.parameters())
    
    model_fine = None
    if args.num_fine_samples_per_ray > 0:
        model_fine = NeRF(num_layers=args.num_layers_fine, 
                         filter_dim=args.num_channels_fine, 
                         input_dim=input_dim, 
                         output_dim=output_dim, 
                         skip_layers=skip_layers, 
                         viewdir_dim=viewdir_dim, 
                         use_viewdirs=args.use_viewdirs).to(device)
        
        grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn, posenc_xyz_fn=posenc_xyz_fn, posenc_dir_fn=posenc_dir_fn, netchunk=args.netchunk)

    # create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    start_itr = 0
    
    # load checkpoints (if stored)
    if args.ckpt_coarsenet_path is not None:
        ckpts_file_lst = [args.ckpt_coarsenet_path]
    else:
        ckpts_file_lst = []
        for file in sorted(os.listdir(os.path.join(args.log_dir, args.expname))):
            if 'tar' in file:
                ckpts_file_lst.append(os.path.join(args.log_dir, args.expname, file))

    print("Found checkpoint files:", ckpts_file_lst)
    if len(ckpts_file_lst) > 0 and args.no_reload_ckpt == False:
        checkpoint_path = ckpts_file_lst[-1]
        print("Reloading stored weights/checkpoints for coarse network from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        start_itr = checkpoint['global_step']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_coarse.load_state_dict(checkpoint['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(checkpoint['network_fine_state_dict'])
    
    render_train_options = {
        'network_query_fn' : network_query_fn,
        'network_fn' : model_coarse,
        'network_fine' : model_fine,
        'num_samples_per_ray' : args.num_samples_per_ray,
        'num_fine_samples_per_ray' : args.num_fine_samples_per_ray,
        'use_viewdirs' : args.use_viewdirs,
        'is_bkgd_white' : args.is_bkgd_white,
        'raw_noise_std' : args.raw_noise_std,
        'perturb' : args.perturb,
    }
    
    render_test_options = {}
    for key in render_train_options:
        render_test_options[key] = render_train_options[key]
    render_test_options['perturb'] = False
    render_test_options['raw_noise_std'] = 0.

    return render_train_options, render_test_options, start_itr, grad_vars, optimizer


def compute_rays(img_height, img_width, cam_int_mat, cam_pose):
    pixel_coord_x, pixel_coord_y = np.meshgrid(np.arange(img_width, dtype=np.float32), np.arange(img_height, dtype=np.float32), indexing='xy')
    
    dirs = np.stack([(pixel_coord_x - cam_int_mat[0][2]) / cam_int_mat[0][0], (pixel_coord_y - cam_int_mat[1][2]) / cam_int_mat[1][1], np.ones_like(pixel_coord_x)], axis=-1)
    
    # Rotate ray directions from camera frame to the world frame
    ray_dirs = np.sum(dirs[..., np.newaxis, :] * cam_pose[:3, :3], axis=-1)
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    ray_origins = np.broadcast_to(cam_pose[:3,-1], np.shape(ray_dirs))
    
    return ray_origins, ray_dirs


def render(img_height, img_width, cam_int_mat, chunk=1024*32, rays=None, cam_pose=None, near_dist=0., far_dist=1.0, **kwargs):
    
    if cam_pose is not None:
        ray_origins, ray_dirs = get_rays(img_height, img_width, cam_int_mat, cam_pose)
    else:
        ray_origins, ray_dirs = rays      # [1, ray_batch_size, 3], [1, ray_batch_size, 3]

    # normalize each direction vector to a unit vector
    viewdirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()     # [ray_batch_size, 3]

    sh = ray_dirs.shape # [1, ray_batch_size, 3]

    # Create ray batch
    ray_origins = torch.reshape(ray_origins, [-1, 3]).float()     # [ray_batch_size, 3]
    ray_dirs = torch.reshape(ray_dirs, [-1, 3]).float()           # [ray_batch_size, 3]

    near_dist_tensor = near_dist * torch.ones_like(ray_dirs[...,:1])   # [ray_batch_size,]
    far_dist_tensor = far_dist * torch.ones_like(ray_dirs[...,:1])     # [ray_batch_size,]
    
    rays = torch.cat([ray_origins, ray_dirs, near_dist_tensor, far_dist_tensor, viewdirs], dim=-1)    # [ray_batch_size, 3+3+2+3=11]

    # Render and reshape
    all_ret = convert_rays_to_batch(rays, chunk, **kwargs)
    
    for key in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[key].shape[1:])
        all_ret[key] = torch.reshape(all_ret[key], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[key] for key in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    
    return ret_list + [ret_dict]


def convert_nerfout_to_rgb(nerf_out, z_vals, ray_dirs, raw_noise_std=0, is_bkgd_white=False):
    """
    Converts raw NeRF output (model's predictions) into RGB and other maps (semantically meaningful)
    Input:
        nerf_out: [num_rays, num_samples along ray, 4] model prediction
        z_vals: [num_rays, num_samples along ray] integration time
        ray_dirs: [num_rays, 3] direction of each ray
    
    Returns:
        rgb_map: [num_rays, 3] estimated rgb color of a ray
        disp_map: [num_rays] disparity map (inverse of depth map)
        acc_map: [num_rays] sum of weights along each ray
        weights: [num_rays, num_samples] weights assigned to each sampled color
        depth_map: [num_rays] estimated distance to object
    """
    
    num_rays = z_vals.shape[0]

    # compute the distance between consecutive elements of points sampled along a ray
    dists = z_vals[..., 1:] - z_vals[..., :-1]       # (num_rays, num_samples-1)
#     dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)   # [N_rays, N_samples]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  
    
    # multiply each distance by the norm of its corresponding direction ray to get real-world distance
    dists = dists * torch.norm(ray_dirs[..., None, :], dim=-1)
#     print(dists.shape)
    
    rgb = torch.sigmoid(nerf_out[...,:3])  # [N_rays, N_samples, 3]
    
    noise = 0
    if raw_noise_std > 0:
        noise = torch.randn(nerf_out[..., 3].shape) * raw_noise_std
        
    alpha = 1.0 - torch.exp(-nn.functional.relu(nerf_out[..., 3] + noise) * dists)  # (num_rays, num_samples)
    
#     term1 = torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=-1)
#     term2 = torch.cumprod(term1, -1)[:, 0:-1]
# #     term2 = term2[..., 0] = 1
#     weights = alpha * term2

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
    
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))
    acc_map = torch.sum(weights, dim=-1)
    
    if is_bkgd_white:
        rgb_map = rgb_map + (1 - acc_map[..., None])
    
    return rgb_map, disp_map, acc_map, weights, depth_map



def stratified_sampling(ray_batch, network_fn, network_query_fn, num_samples_per_ray, perturb=0, 
               num_fine_samples_per_ray=0, network_fine=None, is_bkgd_white=False, raw_noise_std=0, verbose=False, pytest=False):
    num_rays = ray_batch.shape[0]
    ray_origins = ray_batch[:, 0:3]    # (num_rays, 3)
    ray_dirs = ray_batch[:, 3:6]    # (num_rays, 3)
    
    if ray_batch.shape[-1] > 8:
        viewdirs = ray_batch[:, -3:]
    else:
        viewdirs = None
    
    bounds = torch.reshape(ray_batch[..., 6:8], [-1,1,2])
    near_dist, far_dist = bounds[..., 0], bounds[..., 1]
    
    t_vals = torch.linspace(0, 1.0, steps=num_samples_per_ray)
    z_vals = near_dist * (1.0 - t_vals) + far_dist * t_vals
    # z_vals = 1.0 / (1.0/near_dist * (1.0-t_vals) + 1./far_dist * t_vals)
    
    z_vals = z_vals.expand([num_rays, num_samples_per_ray])
    
    if perturb > 0:
        # get intervals between samples
        mid_vals = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        
        upper_vals = torch.cat([mid_vals, z_vals[..., -1:]], dim=-1)
        lower_vals = torch.cat([z_vals[..., :1], mid_vals], dim=-1)
        
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        
        z_vals = lower_vals + (upper_vals - lower_vals) * t_rand
        
    pts = ray_origins[..., None, :] + ray_dirs[..., None, :] * z_vals[..., :, None]  # (num_rays, num_samples, 3)
    
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = convert_nerfout_to_rgb(raw, z_vals, ray_dirs, raw_noise_std, is_bkgd_white, pytest=pytest)
        
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        
    if num_fine_samples_per_ray > 0:
        ret['rgb0'] = rgb_map_copy
        ret['disp0'] = disp_map_copy
        ret['acc0'] = acc_map_copy
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # (num_rays,)
    
    return ret