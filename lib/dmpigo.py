import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter_add, segment_coo

from . import grid
from .act import GaussianActivation
from .dvgo import Raw2Alpha, Alphas2Weights, render_utils_cuda


'''Model'''
class DirectMPIGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 spatial_pe=0,
                 **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = grid.DenseGrid(
                channels=1, world_size=[1,1,mpi_depth],
                xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe, 'spatial_pe': spatial_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(spatial_pe)]))
            self.dim0 = (3+3*viewbase_pe*2+3+3*spatial_pe*2) + self.k0_dim
            self.pe_dim = 3+3*viewbase_pe*2+3+3*spatial_pe*2

            self.dim_rend = 3  # kwargs['dim_rend']
            self.act_type = kwargs['act_type']
            if self.act_type == 'relu':
                act = nn.ReLU(inplace=True)
            elif self.act_type == 'gauss':
                act = GaussianActivation(a=0.05)
            elif self.act_type == 'lkrelu':
                act = nn.LeakyReLU()

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
                # self.rgbper_layer = nn.Sequential(
                #     nn.Linear(self.dim_rend, 3)
                # )
                nn.init.constant_(self.rend_layer[-1].bias, 0)
            else:
                self.rgbnet = nn.Sequential(
                    nn.Linear(self.dim0, rgbnet_width), act,
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), act)
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, self.dim_rend),
                )
                nn.init.constant_(self.rgbnet[-1].bias, 0)

            print('dmpigo: densitye grid', self.density)
            print('dmpigo: feature grid', self.k0)
            self.mode_type = kwargs['mode_type']
            if self.mode_type == 'TRANS':
                self.trans_nn = transformer.Transformer(self.dim0, nhead=3, dim_feedforward=48)
                print('dmpigo: transnet', self.trans_nn)
            elif self.mode_type == 'adain':
                self.adainet = adain.ADANet(input_channels=self.dim0, pos_channels=self.dim0, num_channels=rgbnet_width, output_channels=3, act_type=self.act_type)
                print('dmpigo: adainet', self.adainet)
            else:
                print('dmpigo: mlp', self.rgbnet)

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

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
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
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            dens = self.density.get_dense_grid() + self.act_shift.grid
            self_alpha = F.max_pool3d(self.activate_density(dens), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples, _ = self.sample_ray(
                        rays_o=rays_o.to(device), rays_d=rays_d.to(device), **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print(f'dmpigo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

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
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox

        ray_pts = ray_pts.view(-1, 3)
        ray_pts = ray_pts[mask_inbbox.view(-1)]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples, mask_inbbox

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
        ray_pts, ray_id, step_id, N_samples, mask_inbbox = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask1 = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask1]
            ray_id = ray_id[mask1]
            step_id = step_id[mask1]

        # query for alpha w/ post-activation
        density = self.density(ray_pts) + self.act_shift(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask2 = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask2]
            ray_id = ray_id[mask2]
            step_id = step_id[mask2]
            alpha = alpha[mask2]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask3 = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask3]
            ray_id = ray_id[mask3]
            step_id = step_id[mask3]
            alpha = alpha[mask3]
            weights = weights[mask3]

        # query for color
        vox_emb = self.k0(ray_pts)

        pe_spa = ((ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        # B_gau = torch.normal(mean=0, std=1, size=(128, 3)).to(rays_o.device) * 10
        # pe_emb = input_mapping(pe_spa.detach().clone(), B_gau)

        if self.rgbnet is None:
            # no view-depend effect
            rgb_raw = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            pe_emb = (pe_spa.unsqueeze(-1) * self.posfreq).flatten(-2)
            pe_emb = torch.cat([pe_spa, pe_emb.sin(), pe_emb.cos()], -1)

            if self.mode_type == 'TRANS':
                rgb_feat = torch.cat([vox_emb, pe_emb, viewdirs_emb], -1)
                rgb_feat_pre3 = torch.zeros((mask3.shape[0], self.dim0))
                rgb_feat_pre3[mask3] = rgb_feat
                rgb_feat_pre2 = torch.zeros((mask2.shape[0], self.dim0))
                rgb_feat_pre2[mask2] = rgb_feat_pre3
                rgb_feat_pre1 = torch.zeros((mask1.shape[0], self.dim0))
                rgb_feat_pre1[mask1] = rgb_feat_pre2
                rgb_feat_ori = torch.zeros((mask_inbbox.shape[0], mask_inbbox.shape[1], self.dim0))
                rgb_feat_ori[mask_inbbox] = rgb_feat_pre1
                # rgb_logit_tran = self.reformer(rgb_feat_ori)
                rgb_logit_tran = self.trans_nn(rgb_feat_ori)
                rgb_logit_tran = rgb_logit_tran[mask_inbbox.view(-1)][mask1][mask2][mask3]
                rgb_raw = torch.sigmoid(rgb_logit_tran)
            elif self.mode_type == 'adain':
                rgb_feat = torch.cat([vox_emb, pe_emb, viewdirs_emb], -1)
                # rgb_feat = vox_emb
                # pos_feat = torch.cat([pe_emb, viewdirs_emb], -1)
                rgb_logit = self.adainet(rgb_feat, rgb_feat)
                rgb_raw = torch.sigmoid(rgb_logit)
            else:
                rgb_feat = torch.cat([vox_emb, pe_emb, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)
                if self.dim_rend == 3:
                    rgb_raw = torch.sigmoid(rgb_logit)
                else:
                    rgb_raw = torch.sigmoid(rgb_logit)

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

        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched = rgb_marched + (alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
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
            'n_max': N_samples,
            's': s,
        })
        # print('alphainv_last shape:', alphainv_last.shape)
        # print('weights shape:', weights.shape)
        # print('raw_alpha shape:', alpha.shape)
        # print('raw_rgb shape:', rgb.shape)
        # print('ray_id shape:', ray_id.shape)
        # print('rgb_marched shape:', rgb_marched.shape)
        # print('s shape:', s.shape, '\n')

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
