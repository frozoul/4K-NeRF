import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch_scatter import segment_coo
from . import grid
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)


'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        self.dim_rend = 3  # kwargs['dim_rend']
        self.act_type = 'mlp' # kwargs['act_type']
        self.mode_type = 'mlp'

        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)

            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3

            self.dim0 = dim0
            if self.dim_rend > 3:
                self.rgbnet = nn.Sequential(
                    nn.Linear(self.dim0, rgbnet_width),
                    nn.LeakyReLU(),
                    nn.Linear(rgbnet_width, self.dim_rend),
                    nn.LeakyReLU(),
                )
                self.rend_layer = nn.Sequential(
                    nn.Linear(self.dim_rend, 3),
                )
                nn.init.constant_(self.rend_layer[-1].bias, 0)
            else:
                self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
                nn.init.constant_(self.rgbnet[-1].bias, 0)

            print('dvgo: feature voxel grid', self.k0)
            print('dvgo: mlp', self.rgbnet)


        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.max_world_size = self.world_size.max()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'mode_type': self.mode_type,
            'act_type': self.act_type,
            'dim_rend': self.dim_rend,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu())+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += (ones.grid.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        N_samples = int((self.max_world_size-1)/stepsize) + 1

        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox

        # mask_ori = torch.zeros((len(N_steps), N_steps.max())).bool()
        # for idx, item in enumerate(N_steps):
        #     mask_ori[idx, :item] = mask_inbbox[ray_id == idx]
        mask_ori = None

        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, mask_ori, N_samples

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox, N_samples = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask1 = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask1]
            ray_id = ray_id[mask1]
            step_id = step_id[mask1]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask2 = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask2]
            ray_id = ray_id[mask2]
            step_id = step_id[mask2]
            density = density[mask2]
            alpha = alpha[mask2]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask3 = (weights > self.fast_color_thres)
            weights = weights[mask3]
            alpha = alpha[mask3]
            ray_pts = ray_pts[mask3]
            ray_id = ray_id[mask3]
            step_id = step_id[mask3]

        # query for color
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb_raw = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)

            if self.mode_type == 'TRANS':
                rgb_feat_pre3 = torch.zeros((mask3.shape[0], self.dim0))
                rgb_feat_pre3[mask3] = rgb_feat
                rgb_feat_pre2 = torch.zeros((mask2.shape[0], self.dim0))
                rgb_feat_pre2[mask2] = rgb_feat_pre3
                rgb_feat_pre1 = torch.zeros((mask1.shape[0], self.dim0))
                rgb_feat_pre1[mask1] = rgb_feat_pre2
                rgb_feat_ori = torch.zeros((mask_inbbox.shape[0], mask_inbbox.shape[1], self.dim0))
                rgb_feat_ori[mask_inbbox] = rgb_feat_pre1
                # rgb_logit_tran = self.reformer(rgb_feat_ori)
                rgb_logit = self.trans_nn(rgb_feat_ori)
                rgb_logit = rgb_logit[mask_inbbox.view(-1)][mask1][mask2][mask3]
            elif self.mode_type == 'adain':
                rgb_logit = self.adainet(rgb_feat, rgb_feat)
            else:
                rgb_logit = self.rgbnet(rgb_feat)

            if self.rgbnet_direct:
                rgb_raw = torch.sigmoid(rgb_logit)
            else:
                rgb_raw = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_feature = segment_coo(
                src=(weights.unsqueeze(-1) * rgb_raw),
                index=ray_id,
                out=torch.zeros([N, self.dim_rend]),
                reduce='sum')
        if self.dim_rend > 3:
            rgb_raw = torch.sigmoid(self.rend_layer(rgb_raw))
            rgb_marched = self.rend_layer(rgb_feature)
            # rgb_marched = torch.sigmoid(rgb_marched)
        else:
            rgb_marched = rgb_feature

        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id+0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'rgb_feature': rgb_feature,
            'raw_alpha': alpha,
            'raw_rgb': rgb_raw,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),  # step_id
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
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


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling_sr(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs, cfgs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    H, W = HW[0]
    # rgb_tr = torch.zeros([N,3], device=DEVICE)
    rgb_tr  = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr_ori.device)
    rays_o_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr.device)

    def mask_patch_generator(arr_all, index_all):
        # arr_all_sr = arr_all
        list_bp = list(range(len(arr_all)))
        num_total = len(list_bp)
        idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0

        while True:
            if top >= num_total:
                idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0
            bp_chioce = idx_im[top]
            image_chioce = index_all[bp_chioce]
            patch_chioce = arr_all[bp_chioce]
            patch_4x_chioce = patch_chioce
            # patch_chioce = arr_all[patch_ind]
            # patch_4x_chioce = arr_all_sr[patch_ind]
            top += 1
            pr, pc = patch_chioce.shape[0], patch_chioce.shape[1]
            patch_chioce = patch_chioce.reshape(-1, 2)
            patch_4x_chioce = patch_4x_chioce.reshape(-1, 2)
            patch_chioce = np.moveaxis(patch_chioce, -1, 0)
            patch_4x_chioce = np.moveaxis(patch_4x_chioce, -1, 0)
            yield image_chioce, list(patch_chioce[0]), list(patch_chioce[1]), list(patch_4x_chioce[0]), list(patch_4x_chioce[1]), [pr, pc]

    imsz = []
    top = 0
    idx_b = 0
    patch_all = []
    index_all =[]
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        arr_all = patch_gen(imsz=[H, W], num_im=100, BS=4096, sz_patch=CHUNK)
        masks_arr = [mask[arr[:, :, 0], arr[:, :, 1]] for arr in arr_all]
        for idx, patch_mask in enumerate(masks_arr):
            if patch_mask.sum() > 2048:
                index_all.append(idx_b)
                patch_all.append(arr_all[idx])

        rgb_tr[idx_b].copy_(img)
        rays_o_tr[idx_b].copy_(rays_o.to(DEVICE))
        rays_d_tr[idx_b].copy_(rays_d.to(DEVICE))
        viewdirs_tr[idx_b].copy_(viewdirs.to(DEVICE))
        imsz.append(n)
        top += n
        idx_b += 1
        print(f'patches of image {idx_b} has generated.')

    patch_all = np.stack(patch_all, axis=0)
    index_all = np.stack(index_all, axis=0)
    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    patch_generator = mask_patch_generator(patch_all, index_all)
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, patch_generator


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS


def batch_images_generator(N, imsz, BS):
    idx, top = range(imsz), 0
    n_im = 0
    while True:
        if top + BS >= imsz:
            yield idx[top:imsz], n_im, True
            idx, top = range(imsz), 0
            n_im += 1
            if n_im >= N:
                n_im = 0
        else:
            yield idx[top:top+BS], n_im, False
            top += BS


def simg_patch_indices_generator(imsz, BS):
    BS = BS // 64
    H, W = imsz[0], imsz[1]
    num_x, num_y = H // BS, W // BS
    x = np.linspace(0, W-1, W)
    y = np.linspace(0, H-1, H)
    xx, yy = np.meshgrid(x, y)
    arr_index = np.stack((yy, xx), axis=-1).astype(np.int64)

    slice_x = np.linspace(1, num_x, num_x).astype(np.int64)*BS
    slcie_y = np.linspace(1, num_y, num_y).astype(np.int64)*BS
    arr_yp = np.split(arr_index, slice_x, axis=0)
    arr_yp_last = arr_yp.pop(-1)
    arr_yp = np.stack(arr_yp, axis=-1)
    arr_xyp = np.split(arr_yp, slcie_y, axis=1)
    arr_yp_last = np.split(arr_yp_last, slcie_y, axis=1)
    arr_xp_last = arr_xyp.pop(-1)
    arr_xp_last = list(np.moveaxis(arr_xp_last, -1, 0))
    arr_xyp = np.concatenate(arr_xyp, axis=-1)
    arr_xyp = list(np.moveaxis(arr_xyp, -1, 0))

    arr_all = []
    arr_all.extend(arr_xyp)
    arr_all.extend(arr_xp_last)
    arr_all.extend(arr_yp_last)
    num_p = len(arr_all)
    idx, top = torch.LongTensor(np.random.permutation(num_p)), 0
    while True:
        if top >= num_p:
            idx, top = torch.LongTensor(np.random.permutation(num_p)), 0
        patch_chioce = arr_all[idx[top]]
        top += 1
        patch_chioce = patch_chioce.reshape(-1, 2)
        yield list(np.moveaxis(patch_chioce, -1, 0))


def patch_gen(imsz, num_im, BS, sz_patch):
        BS = BS // sz_patch
        H, W = imsz[0], imsz[1]
        num_x, num_y = H // BS, W // BS
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xx, yy = np.meshgrid(x, y)
        arr_index = np.stack((yy, xx), axis=-1).astype(np.int64)

        slice_x = np.linspace(1, num_x, num_x).astype(np.int64)*BS
        slcie_y = np.linspace(1, num_y, num_y).astype(np.int64)*BS
        arr_yp = np.split(arr_index, slice_x, axis=0)
        arr_yp_last = arr_yp.pop(-1)
        arr_yp = np.stack(arr_yp, axis=-1)
        arr_xyp = np.split(arr_yp, slcie_y, axis=1)
        arr_yp_last = np.split(arr_yp_last, slcie_y, axis=1)
        arr_xp_last = arr_xyp.pop(-1)
        arr_xp_last = list(np.moveaxis(arr_xp_last, -1, 0))
        arr_xyp = np.concatenate(arr_xyp, axis=-1)
        arr_xyp = list(np.moveaxis(arr_xyp, -1, 0))

        arr_all = []
        arr_all.extend(arr_xyp)
        arr_all.extend(arr_xp_last)
        arr_all.extend(arr_yp_last)

        return arr_all

def mimg_patch_indices_generator(imsz, num_im, BS, sz_patch, sr_ratio):
    
    arr_all = patch_gen(imsz, num_im, BS, sz_patch)
    arr_all_sr = patch_gen(imsz*sr_ratio, num_im, BS*sr_ratio, sz_patch)
    
    num_p = len(arr_all)
    list_p = np.ones(num_p)
    list_b = [list_p*i for i in range(num_im)]
    list_b = np.concatenate(list_b, axis=0)
    list_p = np.array(range(num_p))
    list_p = np.tile(list_p, num_im)
    list_bp = np.stack((list_b, list_p), axis=1)
    num_total = num_p*num_im
    idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0

    while True:
        if top >= num_total:
            idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0
        bp_chioce = idx_im[top]
        image_chioce, patch_ind = bp_chioce[0], bp_chioce[1]
        patch_chioce = arr_all[patch_ind]
        patch_4x_chioce = arr_all_sr[patch_ind]
        top += 1
        pr, pc = patch_chioce.shape[0], patch_chioce.shape[1]
        patch_chioce = patch_chioce.reshape(-1, 2)
        patch_4x_chioce = patch_4x_chioce.reshape(-1, 2)
        patch_chioce = np.moveaxis(patch_chioce, -1, 0)
        patch_4x_chioce = np.moveaxis(patch_4x_chioce, -1, 0)
        yield image_chioce, list(patch_chioce[0]), list(patch_chioce[1]), list(patch_4x_chioce[0]), list(patch_4x_chioce[1]), [pr, pc]