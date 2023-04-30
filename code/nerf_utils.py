import os
import sys
import time
from PIL import Image
import imageio
from tqdm import tqdm, trange
import numpy as np

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F


DEBUG = False

# define device type - cuda:0 or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log10(x)


def to8b(x):
    return (255.0 * np.clip(x, 0, 1.0)).astype(np.uint8)


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)
    
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    Prepares inputs and applies network 'fn'.
    Arguments:
        inputs: [num_rays, num_samples_per_ray, 3]
        viewdirs: [num_rays, 3]
        embed_fn: positional encoding function for xyz coordinates
        embeddirs_fn: positional encoding function for direction (d1,d2,d3) coordinates
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [num_rays * num_samples_per_ray, 3]
    embedded_xyz = embed_fn(inputs_flat)                         # [num_rays * num_samples_per_ray, 3 * num_freqs_xyz * 2 + 3]

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)   # [num_rays, num_samples_per_ray, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # [num_rays * num_samples_per_ray, 3]
        embedded_dirs = embeddirs_fn(input_dirs_flat)         # [num_rays * num_samples_per_ray, 3 * num_freqs_dirs * 2 + 3]
        embedded = torch.cat([embedded_xyz, embedded_dirs], dim=-1)  # [num_rays * num_samples_per_ray, (3 * num_freqs_xyz * 2 + 3) + (3 * num_freqs_dirs * 2 + 3)]
        outputs_flat = batchify(fn, netchunk)(embedded)
    else:
        outputs_flat = batchify(fn, netchunk)(embedded_xyz)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """
    Render rays in smaller minibatches to avoid OOM (Out Of Memory error).
    Arguments:
        rays_flat: [ray_batch_size, 3+3+2+3 = 11 (ro+rd+near+far+normalized_viewdir)]
    """
    all_ret = {}
    
    for i in range(0, rays_flat.shape[0], chunk):     # (start, stop, step)
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        
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
        self.embed_fn = [lambda x: x]
        
        if self.include_input:
            self.out_dim = self.input_dim * (2 * self.num_freqs + 1)
        else:
            self.out_dim = self.input_dim * (2 * self.num_freqs)

        if self.log_space:
            self.freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2**(self.num_freqs - 1), self.num_freqs)
            
        # Alternate sin and cos
        for freq in self.freq_bands:
            self.embed_fn.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fn.append(lambda x, freq=freq: torch.cos(x * freq))
    
    def forward(self, x):
        """
        apply positional encoding to input x
        """
#         self.embed_fn = []
#         if self.include_input:
#             self.embed_fn.append(x)
        
#         for freq in self.freq_bands:
#             self.embed_fn.append(torch.sin(x * freq))
#             self.embed_fn.append(torch.cos(x * freq))
        
        return torch.cat([fn(x) for fn in self.embed_fn], dim=-1)


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
        
        # create model layers
        layers = []
        
        for i in range(self.num_layers-1):
            if i in self.skip_layers:
                layers.append(nn.Linear(filter_dim + input_dim, filter_dim))
            else:
                layers.append(nn.Linear(filter_dim, filter_dim))

        self.pts_layers = nn.ModuleList([nn.Linear(self.input_dim, filter_dim)] + layers)
        
        # bottleneck layers
        if self.use_viewdirs:
            # if using view directions, split sigma and RGB
            self.sigma_layer = nn.Linear(filter_dim, 1)
            self.feature_layer = nn.Linear(filter_dim, filter_dim)
            self.viewdir_layers = nn.ModuleList([nn.Linear(filter_dim + self.viewdir_dim, filter_dim // 2)])
            self.rgb_layer = nn.Linear(filter_dim // 2, 3)
        else:
            # if no viewing directions, use simpler output
            self.output_layer = nn.Linear(filter_dim, output_dim)
        

    def forward(self, x):
#         print("Hola Saqib Babu:", x.shape, self.input_dim, self.viewdir_dim)
        input_pts, viewdir = torch.split(x, [self.input_dim, self.viewdir_dim], dim=-1)
#         print("Here Saqib:", x.shape, input_pts.shape, viewdir.shape)
        
#         if (self.viewdir_dim is not None and viewdir is None) or (self.viewdir_dim is None and viewdir is not None):
#             raise ValueError("Mistake in input view direction !!!")
        
        y = torch.clone(input_pts)
        
        for i, layer in enumerate(self.pts_layers):
            y = self.relu_layer(layer(y))
            
            if i in self.skip_layers:
                y = torch.cat([input_pts, y], dim=-1)
                
        # apply bottleneck
        if self.use_viewdirs:
            # split sigma from the network output
            sigma = self.relu_layer(self.sigma_layer(y))
            
            # pass through bottleneck to get RGB
            feature = self.feature_layer(y)
            y = torch.cat([feature, viewdir], dim=-1)
            
            for j, layer in enumerate(self.viewdir_layers):
                y = self.relu_layer(layer(y))  
            
            rgb = self.rgb_layer(y)
        
            # concatenate sigma to rgb output
            output = torch.cat([rgb, sigma], dim=-1)
        else:
            # simple output
            output = self.output_layer(y)
        
        return output


def create_NeRF(args):
    """
    Instantiate NeRF's MLP model
    """
    
    # get positional encoding for the position (x,y,z) coordinates
    if args.i_embed == -1:
        embed_fn, input_dim = nn.Identity(), 3
    else:
        embedder_obj = PositionalEncoder(input_dim=3, num_freqs=args.num_freqs_xyz, log_space=True, include_input=True)
        embed_fn = lambda x, eo=embedder_obj : eo(x)
        input_dim = embedder_obj.out_dim
    
    viewdir_dim = 0
    embeddirs_fn = None
    
    # get positional encoding for the view direction (d1, d2, d3 / theta, phi) coordinates
    if args.use_viewdirs:
        if args.i_embed == -1:
            embeddirs_fn, viewdir_dim = nn.Identity(), 3
        else:
            embedder_obj = PositionalEncoder(input_dim=3, num_freqs=args.num_freqs_viewdir, log_space=True, include_input=True)
            embeddirs_fn = lambda x, eo=embedder_obj : eo(x)
            viewdir_dim = embedder_obj.out_dim
    
    output_dim = 5 if args.num_fine_samples_per_ray > 0 else 4
    skip_layers = [args.netdepth//2 - 1]
    
    model_coarse = NeRF(num_layers=args.netdepth, 
                     filter_dim=args.netwidth, 
                     input_dim=input_dim, 
                     output_dim=output_dim, 
                     skip_layers=skip_layers, 
                     viewdir_dim=viewdir_dim, 
                     use_viewdirs=args.use_viewdirs).to(device)
    
    grad_vars = list(model_coarse.parameters())
    
    model_fine = None
    if args.num_fine_samples_per_ray > 0:
        model_fine = NeRF(num_layers=args.netdepth_fine, 
                         filter_dim=args.netwidth_fine, 
                         input_dim=input_dim, 
                         output_dim=output_dim, 
                         skip_layers=skip_layers, 
                         viewdir_dim=viewdir_dim, 
                         use_viewdirs=args.use_viewdirs).to(device)
        
        grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                         embed_fn=embed_fn,
                                                                         embeddirs_fn=embeddirs_fn,
                                                                         netchunk=args.netchunk)

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
        # load model
        model_coarse.load_state_dict(checkpoint['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(checkpoint['network_fine_state_dict'])
    
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_fn' : model_coarse,
        'network_fine' : model_fine,
        'num_samples_per_ray' : args.num_samples_per_ray,
        'num_fine_samples_per_ray' : args.num_fine_samples_per_ray,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'perturb' : args.perturb,
    }
    
    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    
    render_kwargs_test = {key : render_kwargs_train[key] for key in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start_itr, grad_vars, optimizer


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    print("WHAT THE FUCK !!!")
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    print("REALLY WHAT THE FUCK !!!")
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, near=0., far=1.0, ndc=True, 
           use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays      # [1, ray_batch_size, 3], [1, ray_batch_size, 3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        
        # normalize each direction vector to a unit vector
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()     # [ray_batch_size, 3]

    sh = rays_d.shape # [1, ray_batch_size, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()     # [ray_batch_size, 3]
    rays_d = torch.reshape(rays_d, [-1,3]).float()     # [ray_batch_size, 3]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])  # [ray_batch_size,], [ray_batch_size,]
    
    rays = torch.cat([rays_o, rays_d, near, far], -1)    # [ray_batch_size, 3+3+2=8]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)       # [ray_batch_size, 8+3 = 11]

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    
    for key in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[key].shape[1:])
        all_ret[key] = torch.reshape(all_ret[key], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[key] for key in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    
    return ret_list + [ret_dict]


def render_path(render_poses, H, W, focal, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

#         if gt_imgs is not None and render_factor==0:
#             p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
#             print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def raw2outputs(nerf_out, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    Converts raw NeRF output (model's predictions) into RGB and other maps (semantically meaningful)
    Input:
        nerf_out: [num_rays, num_samples along ray, 4] model prediction
        z_vals: [num_rays, num_samples along ray] integration time
        rays_d: [num_rays, 3] direction of each ray
    
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
#     dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    
    # multiply each distance by the norm of its corresponding direction ray to get real-world distance
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
#     print(dists.shape)
    
    rgb = torch.sigmoid(nerf_out[...,:3])  # [N_rays, N_samples, 3]
    noise = 0
    if raw_noise_std > 0:
        noise = torch.randn(nerf_out[..., 3].shape) * raw_noise_std
        
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(nerf_out[...,3].shape)) * raw_noise_std
            noise = torch.tensor(noise)
        
    alpha = 1.0 - torch.exp(-nn.functional.relu(nerf_out[..., 3] + noise) * dists)  # (num_rays, num_samples)
    
#     term1 = torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=-1)
#     term2 = torch.cumprod(term1, -1)[:, 0:-1]
# #     term2 = term2[..., 0] = 1
#     weights = alpha * term2

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1 - acc_map[..., None])
    
    return rgb_map, disp_map, acc_map, weights, depth_map



def render_rays(ray_batch, network_fn, network_query_fn, num_samples_per_ray, retraw=False, lindisp=False, perturb=0, 
               num_fine_samples_per_ray=0, network_fine=None, white_bkgd=False, raw_noise_std=0, verbose=False, pytest=False):
    """
    Volumetric rendering
    ray_batch: array of shape [batch_size, ...]. all information necessary for sampling along a ray including ray 
                origin, ray direction, min dist, max dist, and viewing direction (unit magnitude)
    network_fn: model for predicting RGB and density at each point
    network_query_fn: function used for passing queries to network_fn
    num_samples_per_ray: number of different samples along ray
    retraw: If True, include model's raw unprocessed predictions
    linedisp: If True, sample linearly in inverse depth rather than in depth
    perturb: 0/1, If non-zero, then each ray is sampled at stratified random pts in time
    num_fine_samples_per_ray: number of additional times to sample along each ray
    network_fine: "fine" network with same specs as network_fn
    white_bkgd: If True, assume a white background
    raw_noise_std: 
    """
    num_rays = ray_batch.shape[0]
    rays_o = ray_batch[:, 0:3]    # (num_rays, 3)
    rays_d = ray_batch[:, 3:6]    # (num_rays, 3)
    
    if ray_batch.shape[-1] > 8:
        viewdirs = ray_batch[:, -3:]
    else:
        viewdirs = None
    
    bounds = torch.reshape(ray_batch[..., 6:8], [-1,1,2])
    near_dist, far_dist = bounds[..., 0], bounds[..., 1]
    
    t_vals = torch.linspace(0, 1.0, steps=num_samples_per_ray)
    
    if not lindisp:
        z_vals = near_dist * (1.0 - t_vals) + far_dist * t_vals
    else:
        z_vals = 1.0 / (1.0/near_dist * (1.0-t_vals) + 1./far_dist * t_vals)
    
    z_vals = z_vals.expand([num_rays, num_samples_per_ray])
    
    if perturb > 0:
        # get intervals between samples
        mid_vals = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        
        upper_vals = torch.cat([mid_vals, z_vals[..., -1:]], dim=-1)
        lower_vals = torch.cat([z_vals[..., :1], mid_vals], dim=-1)
        
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        
        z_vals = lower_vals + (upper_vals - lower_vals) * t_rand
        
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (num_rays, num_samples, 3)
    
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    if num_fine_samples_per_ray > 0:
        rgb_map_copy = rgb_map
        disp_map_copy = disp_map
        acc_map_copy = acc_map
        
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], num_fine_samples_per_ray, det=(perturb==0), pytest=pytest)
        z_samples = z_samples.detach()
        
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (num_rays, num_samples+num_fine_samples_per_ray, 3)
        
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
        
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
        
    if num_fine_samples_per_ray > 0:
        ret['rgb0'] = rgb_map_copy
        ret['disp0'] = disp_map_copy
        ret['acc0'] = acc_map_copy
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # (num_rays,)
        
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print("Numerical error, contains nan or error")
    
    return ret