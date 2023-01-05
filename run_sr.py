import os, sys, copy, time, random, argparse
from tqdm import tqdm, trange
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import mmcv
import imageio
import numpy as np
from cv2 import resize

import torch
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo, sr_esrnet, sr_unetdisc
from lib.load_data import load_data
from lib.masked_adam import MaskedAdam

from torch_efficient_distloss import flatten_eff_distloss
from torch.utils.tensorboard import SummaryWriter
from basicsr.losses import GANLoss, PerceptualLoss

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--dv_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--sr_path", type=str, default='',
                        help='specific weights file to reload for super resolution network in test stage')
    parser.add_argument("--ftsr_path", type=str, default='',
                        help='specific weights file to reload for super resolution network for finetine')
    parser.add_argument("--ftdvcoa_path", type=str, default='',
                        help='specific weights file to reload for super resolution network for finetine')
    parser.add_argument("--ftdv_path", type=str, default='',
                        help='specific weights file to reload for super resolution network for finetine')
    parser.add_argument("--test_tile", type=int, default=510,
                        help='tile images in test stage to reduce GPU memory cost')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_val",   type=int, default=30000,
                        help='frequency of test training')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, global_step=0, arr_index=None, img_enc=None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    rgb_features = []
    depths = []
    bgmaps = []
    psnrs = []
    viewdirs_all = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'rgb_feature']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        time_rdstart = time.time()
        if arr_index is not None:
            arr_index = torch.from_numpy(arr_index) // 2
            arr_index = arr_index.split(8192, 0)
            img_enc = [img_enc[ind[:, 0], ind[:, 1], :, :] for ind in arr_index]
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, imc, **render_kwargs).items() if k in keys}
                for ro, rd, vd, imc in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0), img_enc)
            ]
        else:
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
            ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        print(f'render time is: {time.time() - time_rdstart}')
        rgb = render_result['rgb_marched'].clamp(0, 1).cpu().numpy()
        rgb_feature = render_result['rgb_feature'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        rgb_features.append(rgb_feature)
        depths.append(depth)
        bgmaps.append(bgmap)
        viewdirs_all.append(viewdirs)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'e{}_{:03d}.png'.format(global_step, i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    rgb_features = np.array(rgb_features)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps, psnrs, viewdirs_all, rgb_features


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    if cfg.data.load_sr:
        kept_keys.add('srgt')
        kept_keys.add('w2c')
        data_dict['srgt'] = torch.FloatTensor(data_dict['srgt'], device='cpu')
        data_dict['w2c'] = torch.FloatTensor(data_dict['w2c'], device='cpu')
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            # viewbase_pe=model_kwargs.viewbase_pe,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ftdv_path:
        reload_ckpt_path = args.ftdv_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)

    # torch.save({
    #         'model_state_dict': model.state_dict()
    #     }, os.path.join(cfg.basedir, cfg.expname, f'{stage}_testsize.tar'))

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    render_viewpoints_kwargs = {
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': cfg_model.stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg_train.ray_sampler == 'patch_simg':
            index_generator = dvgo.simg_patch_indices_generator(HW[0], cfg_train.N_rand)
        elif cfg_train.ray_sampler == 'patch_mimg':
            index_generator = dvgo.mimg_patch_indices_generator(HW[0], len(i_train), cfg_train.N_rand)
        else:
            index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'patch_simg':
            sel_b, sel_r, sel_c = batch_index_sampler()
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'patch_mimg':
            sel_b, sel_r, sel_c = batch_index_sampler()
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

            # if param_group['kname'] == 'rgbnet' and global_step < 5000:
            #     param_group['lr'] = 0

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            summary_writer.add_scalar('train/loss',
                            loss.item(), global_step=global_step)
            summary_writer.add_scalar('train/psnr',
                            np.mean(psnr_lst), global_step=global_step)
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                summary_writer.add_scalar('train/{}'.format(param_group['kname']),
                                param_group['lr'], global_step=global_step)
            psnr_lst = []
        
        # test current step
        if global_step%args.i_val==0:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, 'render_val')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            rgbs, depths, bgmaps, psnrs_test = render_viewpoints(
                    model=model,
                    render_poses=data_dict['poses'][data_dict['i_val']],
                    HW=data_dict['HW'][data_dict['i_val']],
                    Ks=data_dict['Ks'][data_dict['i_val']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_val']],
                    savedir=testsavedir, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    global_step=global_step,
                    **render_viewpoints_kwargs)
            summary_writer.add_scalar('val/psnr',
                            np.mean(psnrs_test), global_step=global_step)
            for idx, rgbtest in enumerate(rgbs):
                summary_writer.add_image(f'val/image_{idx:04d}', np.clip(rgbtest, 0, 1), global_step=global_step, dataformats='HWC')

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)



def scene_rep_reconstruction_sr_patch(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, srgt, w2c = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'srgt', 'w2c'
        ]
    ]
    sr_ratio = int(cfg.data.factor / cfg.data.load_sr)

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ftdv_path:
        reload_ckpt_path = args.ftdv_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)
    model = model.to(device)
    net_sr = sr_esrnet.SFTNet(n_in_colors=cfg_model.dim_rend, scale=sr_ratio, num_feat=64, num_block=5, num_grow_ch=32, num_cond=cfg_model.num_cond, dswise=False).to(device)
    net_sr.load_network(load_path=args.ftsr_path, device=device, strict=False)

    param_sr = []
    param_sr.append({'params': net_sr.parameters(), 'lr': cfg_train.lrate_srnet, 'kname': 'srnet', 'skip_zero_grad': (False)})
    optimizer_sr = MaskedAdam(param_sr)
    print(f'create_optimizer_or_freeze_model: param srnet lr {cfg_train.lrate_srnet}')

    if cfg_train.weight_pcp > 0:
        pcplayer_weight = {
            'conv1_2': 0,
            'conv2_2': 0,
            'conv3_4': 1,
            'conv4_4': 1,
            'conv5_4': 1
        }
        cri_perceptual = PerceptualLoss(layer_weights=pcplayer_weight, vgg_type='vgg19', perceptual_weight=cfg_train.weight_pcp, style_weight=cfg_train.weight_style).to(device)
    if cfg_train.weight_gan > 0:
        cri_gan = GANLoss(gan_type='vanilla', loss_weight=cfg_train.weight_gan)
        if cfg_model.d_model == 'Unet':
            net_d = sr_unetdisc.UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(device)
        elif cfg_model.d_model == 'Unet_pose':
            net_d = sr_unetdisc.UNetDiscriminatorSN_pose(num_in_ch=3, reso=cfg_train.N_patch, c_dim=9, cmap_dim=32, num_feat=64, skip_connection=True).to(device)
        elif cfg_model.d_model == 'Unet_viewdir':
            net_d = sr_unetdisc.UNetDiscriminatorSN_viewdir(num_in_ch=3, reso=cfg_train.N_patch, c_dim=63, cmap_dim=64, num_feat=64, skip_connection=True).to(device)
        param_d = []
        param_d.append({'params': net_d.parameters(), 'lr': cfg_train.lrate_srnet, 'kname': 'd', 'skip_zero_grad': (False)})
        optimizer_d = MaskedAdam(param_d)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': True,
    }
    render_viewpoints_kwargs = {
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': cfg_model.stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }
    
    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'patch_inmask':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, index_generator_mask = dvgo.get_training_rays_in_maskcache_sampling_sr(rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs, cfgs=cfg)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg_train.ray_sampler == 'patch_simg':
            index_generator = dvgo.simg_patch_indices_generator(HW[0], cfg_train.N_rand)
        elif cfg_train.ray_sampler == 'patch_mimg':
            index_generator = dvgo.mimg_patch_indices_generator(HW[0], len(i_train), cfg_train.N_rand, cfg_train.N_patch, sr_ratio)
        elif cfg_train.ray_sampler == 'patch_inmask':
            index_generator = index_generator_mask
        else:
            index_generator = dvgo.batch_images_generator(len(rgb_tr), rgb_tr.shape[1]*rgb_tr.shape[2], cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    rgb_srgt_train = srgt[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
    rgb_srgt_val = srgt[i_val].squeeze()
    if cfg.data.dataset_type == 'llff':
        rgb_srgt_train = rgb_srgt_train.movedim(1, -1)
        rgb_srgt_val = rgb_srgt_val.movedim(0, -1)
        rgb_srgt_val = rgb_srgt_val.unsqueeze(0)
    rgb_srgt_val = rgb_srgt_val.numpy()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_sr_lst = []
    loss_dict = {}
    time0 = time.time()
    global_step = -1
    pout_render = []
    # weights = []
    # raw_alpha = []
    raw_rgb = []
    ray_id = []
    n_max = []
    s = []
    lpips_pre = 1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i, sel_b, im_finish = batch_index_sampler()
            target = rgb_tr[sel_b, sel_i, ...]
            rays_o = rays_o_tr[sel_b, sel_i, ...]
            rays_d = rays_d_tr[sel_b, sel_i, ...]
            viewdirs = viewdirs_tr[sel_b, sel_i, ...]
        elif cfg_train.ray_sampler == 'patch_mimg':
            sel_b, sel_r, sel_c, sel_r_4x, sel_c_4x, ps = batch_index_sampler()
            target = rgb_tr[sel_b, sel_r, sel_c, :]
            target_4x = rgb_srgt_train[sel_b, sel_r_4x, sel_c_4x, :]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            pr, pc = ps[0], ps[1]
        elif cfg_train.ray_sampler == 'patch_inmask':
            sel_b, sel_r, sel_c, sel_r_4x, sel_c_4x, ps = batch_index_sampler()
            target = rgb_tr[sel_b, sel_r, sel_c, :]
            target_4x = rgb_srgt_train[sel_b, sel_r_4x, sel_c_4x, :]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            pr, pc = ps[0], ps[1]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
        
        if cfg_model.d_model == 'Unet_pose':
            w2c_cur = w2c_train[sel_b:sel_b+1, ...]
        elif cfg_model.d_model == 'Unet_viewdir':
            viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
            dis_cond = (viewdirs.unsqueeze(-1) * viewfreq).flatten(-2)
            dis_cond = torch.cat([viewdirs, dis_cond.sin(), dis_cond.cos()], -1)
            dis_cond = dis_cond.reshape(1, pr, pc, -1).movedim(-1, 1).detach()

        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)

        rgb_render = render_result['rgb_feature']
        rgb_march = render_result['rgb_marched']
        rgb_cache = rgb_render.reshape(1, pr, pc, -1).movedim(-1, 1)
        rgb_march = rgb_march.reshape(1, pr, pc, -1).movedim(-1, 1).clamp(0, 1)
        pout_render = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
        loss_total = 0

        # print(f'rgb render size is: {rgb_render.size()}')
         # print(f'rgb marched size is: {rgb_march.size()}')

        if cfg_model.dim_rend == 3:
            loss_pho = cfg_train.weight_main * F.l1_loss(rgb_render, target)
        else:
            loss_pho = cfg_train.weight_main * F.l1_loss(render_result['rgb_marched'], target)

        loss_total += loss_pho
        if 'loss_photo' in loss_dict:
            loss_dict['loss_photo'].append(loss_pho.item())
        else:
            loss_dict['loss_photo'] = []

        if cfg_model.num_cond == 1:
            # use depth as SFT conditional input
            input_cond = render_result['depth']
            input_cond = input_cond.reshape(1, pr, pc, 1).movedim(-1, 1)
        elif cfg_model.num_cond == 63:
            # use viewdir embeding as SFT conditional input
            viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
            input_cond = (viewdirs.unsqueeze(-1) * viewfreq).flatten(-2)
            input_cond = torch.cat([viewdirs, input_cond.sin(), input_cond.cos()], -1)
            input_cond = input_cond.reshape(1, pr, pc, -1).movedim(-1, 1).detach()
        elif cfg_model.num_cond == 64:
            input_cond1 = render_result['depth']
            input_cond1 = input_cond1.reshape(1, pr, pc, 1).movedim(-1, 1).detach()
            viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
            input_cond2 = (viewdirs.unsqueeze(-1) * viewfreq).flatten(-2)
            input_cond2 = torch.cat([viewdirs, input_cond2.sin(), input_cond2.cos()], -1)
            input_cond2 = input_cond2.reshape(1, pr, pc, -1).movedim(-1, 1).detach()
            input_cond = torch.cat((input_cond1, input_cond2), dim=1)

        # if cfg_model.dim_rend == 3:
        #     rgb_sr = net_sr(rgb_march, input_cond)
        # else:
        #     rgb_sr = net_sr(rgb_march, input_cond, rgb_cache)
        rgb_sr = net_sr(rgb_cache, input_cond)

        if cfg_train.weight_gan > 0:
            for p in net_d.parameters():
                p.requires_grad = False

        rgb_hr = target_4x.detach().reshape(sr_ratio*pr, sr_ratio*pc, 3).movedim(-1, 0).unsqueeze(0)

        loss_sr = F.l1_loss(rgb_sr, rgb_hr)
        psnr_sr = utils.mse2psnr((rgb_sr.detach().clamp(0, 1) - rgb_hr).pow(2).mean())

        loss_total += loss_sr
        if 'loss_l1' in loss_dict:
            loss_dict['loss_l1'].append(loss_sr.item())
        else:
            loss_dict['loss_l1'] = []
        if cfg_train.weight_pcp > 0:
            loss_pcp, loss_style = cri_perceptual(rgb_sr, rgb_hr)
            loss_total += loss_pcp
            loss_total += loss_style
            if 'loss_pcp' in loss_dict:
                loss_dict['loss_pcp'].append(loss_pcp.item())
            else:
                loss_dict['loss_pcp'] = []
            if 'loss_style' in loss_dict:
                loss_dict['loss_style'].append(loss_style.item())
            else:
                loss_dict['loss_style'] = []
        if cfg_train.weight_gan > 0:
            if cfg_model.d_model == 'Unet_pose':
                fake_g = net_d(rgb_sr, w2c_cur)
            elif cfg_model.d_model == 'Unet_viewdir':
                fake_g = net_d(rgb_sr, dis_cond)
            else:
                fake_g = net_d(rgb_sr)
            loss_g = cri_gan(fake_g, True, is_disc=False)
            loss_total += loss_g
            if 'loss_g' in loss_dict:
                loss_dict['loss_g'].append(loss_g.item())
            else:
                loss_dict['loss_g'] = []

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        optimizer_sr.zero_grad(set_to_none=True)
        if cfg_train.weight_entropy_last > 0:
            entropy_last_loss = -(pout_render*torch.log(pout_render) + (1-pout_render)*torch.log(1-pout_render)).mean() * cfg_train.weight_entropy_last
            loss_total += entropy_last_loss
            if 'loss_entrp_last' in loss_dict:
                loss_dict['loss_entrp_last'].append(entropy_last_loss.item())
            else:
                loss_dict['loss_entrp_last'] = []
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss_total += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']

            # print('n_max shape:', n_max)
            # print('w shape:', w.shape)
            # print('s shape:', s.shape)
            # print('ray_id shape:', ray_id.shape)

            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss_total += cfg_train.weight_distortion * loss_distortion
            if 'loss_distor' in loss_dict:
                loss_dict['loss_distor'].append(cfg_train.weight_distortion * loss_distortion.item())
            else:
                loss_dict['loss_distor'] = []
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss_total += cfg_train.weight_rgbper * rgbper_loss
            if 'loss_rgbper' in loss_dict:
                loss_dict['loss_rgbper'].append(cfg_train.weight_rgbper * rgbper_loss.item())
            else:
                loss_dict['loss_rgbper'] = []

        loss_total.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o_tr), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o_tr), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        optimizer_sr.step()
        psnr_sr_lst.append(psnr_sr.item())

        if cfg_train.weight_gan > 0:
            for p in net_d.parameters():
                p.requires_grad = True
            optimizer_d.zero_grad()

            if cfg_model.d_model == 'Unet_pose':
                real_d = net_d(rgb_hr, w2c_cur)
            elif cfg_model.d_model == 'Unet_viewdir':
                real_d = net_d(rgb_hr, dis_cond)
            else:
                real_d = net_d(rgb_hr)
            loss_d_real = cri_gan(real_d, True, is_disc=True)
            if 'loss_d_real' in loss_dict:
                loss_dict['loss_d_real'].append(loss_d_real.item())
            else:
                loss_dict['loss_d_real'] = []
            loss_d_real.backward()

            if cfg_model.d_model == 'Unet_pose':
                fake_d = net_d(rgb_sr.detach().clone(), w2c_cur)
            elif cfg_model.d_model == 'Unet_viewdir':
                fake_d = net_d(rgb_sr.detach().clone(), dis_cond)
            else:
                fake_d = net_d(rgb_sr.detach().clone())
            loss_d_fake = cri_gan(fake_d, False, is_disc=True)
            if 'loss_d_fake' in loss_dict:
                loss_dict['loss_d_fake'].append(loss_d_fake.item())
            else:
                loss_dict['loss_d_fake'] = []
            loss_d_fake.backward()
            optimizer_d.step()

        torch.cuda.empty_cache()

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        for i_opt_sr, param_sr in enumerate(optimizer_sr.param_groups):
            param_sr['lr'] = param_sr['lr'] * decay_factor
        
        for i_opt_sr, param_d in enumerate(optimizer_d.param_groups):
            param_d['lr'] = param_d['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm_info = 'scene_rep_reconstruction ({}): img {} iter {:6d} / '.format(stage, sel_b, global_step)
            for dname, dvalue in loss_dict.items():
                tqdm_info += '{}: {:.9f} / '.format(dname, np.mean(dvalue))
            tqdm_info += 'PSNR_SR: {:5.2f} / Eps: {}'.format(np.mean(psnr_sr_lst), eps_time_str)
            tqdm.write(tqdm_info)

            for dname, dvalue in loss_dict.items():
                summary_writer.add_scalar(f'train/{dname}', np.mean(dvalue), global_step=global_step)
                loss_dict[dname] = []
            summary_writer.add_scalar('train/psnr_sr',
                            np.mean(psnr_sr_lst), global_step=global_step)
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                summary_writer.add_scalar('train/{}'.format(param_group['kname']),
                                param_group['lr'], global_step=global_step)
            psnr_sr_lst = []
        
        # test current step
        if global_step%args.i_val==0 or global_step==30001:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, 'render_val')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            with torch.no_grad():
                rgbs, depths, bgmaps, psnrs_test, viewdirs, rgb_features = render_viewpoints(
                        model=model,
                        render_poses=data_dict['poses'][data_dict['i_val']],
                        HW=data_dict['HW'][data_dict['i_val']],
                        Ks=data_dict['Ks'][data_dict['i_val']],
                        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_val']],
                        savedir=testsavedir, dump_images=args.dump_images,
                        eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                        global_step=global_step,
                        **render_viewpoints_kwargs)
                summary_writer.add_scalar('val/psnr_ori', np.mean(psnrs_test), global_step=global_step)
                for idx, rgbsave in enumerate(rgb_features):
                    # summary_writer.add_image(f'val/image_{idx:04d}', np.clip(rgbsave, 0, 1), global_step=global_step, dataformats='HWC')
                    rgbtest = torch.from_numpy(rgbsave).movedim(-1, 0).unsqueeze(0).to(device)
                    rgb = torch.from_numpy(rgbs[idx]).movedim(-1, 0).unsqueeze(0).to(device)
                    
                    if cfg_model.num_cond == 1:
                        input_cond = torch.from_numpy(depths[idx]).movedim(-1, 0).to(device)
                    elif cfg_model.num_cond == 63:
                        HW = data_dict['HW'][data_dict['i_val']].astype(int)[idx]
                        viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
                        input_cond = (viewdirs[idx].unsqueeze(-1) * viewfreq).flatten(-2)
                        input_cond = torch.cat([viewdirs[idx], input_cond.sin(), input_cond.cos()], -1)
                        input_cond = input_cond.reshape(1, HW[0], HW[1], -1).movedim(-1, 1).detach()
                    elif cfg_model.num_cond == 64:
                        input_cond1 = torch.from_numpy(depths[idx]).movedim(-1, 0).to(device)
                        HW = data_dict['HW'][data_dict['i_val']].astype(int)[idx]
                        viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
                        input_cond2 = (viewdirs[idx].unsqueeze(-1) * viewfreq).flatten(-2)
                        input_cond2 = torch.cat([viewdirs[idx], input_cond2.sin(), input_cond2.cos()], -1)
                        input_cond2 = input_cond2.reshape(1, HW[0], HW[1], -1).movedim(-1, 1).detach()
                        input_cond = torch.cat((input_cond1, input_cond2), dim=1)

                    net_sr.eval()
                    if args.test_tile:
                        rgb_srtest = net_sr.tile_process(rgbtest, input_cond, tile_size=args.test_tile)
                    else:
                        rgb_srtest = net_sr(rgbtest, input_cond).detach()

                    imageio.imwrite(os.path.join(testsavedir, f'test_{global_step}.png'), utils.to8b(rgbs[idx]))
                    net_sr.train()

                    rgb_srsave = rgb_srtest.squeeze().movedim(0, -1).detach().clamp(0, 1).cpu().numpy()
                    sr_mse = np.mean(np.square(rgb_srsave - rgb_srgt_val[idx]))
                    sr_psnr = -10. * np.log10(sr_mse)
                    sr_ssim = utils.rgb_ssim(rgb_srsave, rgb_srgt_val[idx], max_val=1)
                    if cfg.data.load_sr == 1:
                        sr_lpips = utils.rgb_lpips(resize(rgb_srsave, dsize=None, fx=1/4, fy=1/4), resize(rgb_srgt_val[idx], dsize=None, fx=1/4, fy=1/4), net_name='vgg', device=device)
                    else:
                        sr_lpips = utils.rgb_lpips(rgb_srsave, rgb_srgt_val[idx], net_name='vgg', device=device)

                    print('Testing psnr', sr_psnr, '(sr)')
                    print('Testing ssim', sr_ssim, '(sr)')
                    print('Testing lpips', sr_lpips, '(sr)')
                    summary_writer.add_scalar('val/psnr_sr', sr_psnr, global_step=global_step)
                    summary_writer.add_scalar('val/ssim_sr', sr_ssim, global_step=global_step)
                    summary_writer.add_scalar('val/lpips_sr', sr_lpips, global_step=global_step)
                    imageio.imwrite(os.path.join(testsavedir, f'testsr_{global_step}_{sel_b}.png'), utils.to8b(rgb_srsave))
            torch.cuda.empty_cache()
        
            # net_sr.save_network(testsavedir, 'sresrnet', global_step)
            if sr_lpips < lpips_pre:
                lpips_pre = sr_lpips
                path_lpips = os.path.join(testsavedir, f'lpips_dvgo.tar')
                torch.save({
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(), }, path_lpips)
                net_sr.save_network(testsavedir, 'sresrnet', -1)

            print(f'scene_rep_reconstruction ({stage}): saved srnet at', testsavedir)

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, 'ckpt_saved')
            os.makedirs(path, exist_ok=True)
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(path, f'{stage}_{global_step:06d}.tar'))
            net_sr.save_network(path, 'sresrnet', global_step)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0:
        # scene_rep_reconstruction(
        #         args=args, cfg=cfg,
        #         cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
        #         xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
        #         data_dict=data_dict, stage='coarse')
        # eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = args.ftdvcoa_path
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction_sr_patch(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    tb_dir = os.path.join(cfg.basedir, cfg.expname)
    summary_writer = SummaryWriter(tb_dir)
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)
    sr_ratio = int(cfg.data.factor / cfg.data.load_sr)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ftdv_path:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        else:
            ckpt_path = args.dv_path

        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, _, _, _ = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, _, _, rgb_features = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)

        net_sr = sr_esrnet.SFTNet(n_in_colors=cfg.fine_model_and_render.dim_rend, scale=sr_ratio, num_feat=64, num_block=5, num_grow_ch=32, num_cond=cfg.fine_model_and_render.num_cond, dswise=False).to(device)
        if args.sr_path:
            net_sr.load_network(load_path=args.sr_path, device=device)
        else:
            sr_load_path = os.path.join(cfg.basedir, cfg.expname, 'render_val/sresrnet_latest.pth')
            net_sr.load_network(load_path=sr_load_path, device=device)
        net_sr.eval()
        rgbsr = []
        for idx, rgbsave in enumerate(tqdm(rgb_features)):
            rgbtest = torch.from_numpy(rgbsave).movedim(-1, 0).unsqueeze(0).to(device)
            rgb = torch.from_numpy(rgbs[idx]).movedim(-1, 0).unsqueeze(0).to(device)

            if cfg.fine_model_and_render.num_cond == 1:
                input_cond = torch.from_numpy(depths).movedim(-1, 1)
                input_cond = input_cond[idx, :, :, :].to(device)
            elif cfg.fine_model_and_render.num_cond == 63:
                HW = data_dict['HW'][data_dict['i_test']].astype(int)[0]
                viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
                input_cond = (viewdirs[idx].unsqueeze(-1) * viewfreq).flatten(-2)
                input_cond = torch.cat([viewdirs[idx], input_cond.sin(), input_cond.cos()], -1)
                input_cond = input_cond.reshape(1, HW[0], HW[1], -1).movedim(-1, 1).detach()
            elif cfg.fine_model_and_render.num_cond == 64:
                input_cond1 = torch.from_numpy(depths[idx]).movedim(-1, 0)
                input_cond1 = input_cond1[:, idx:idx+1, :, :].to(device)
                HW = data_dict['HW'][data_dict['i_test']].astype(int)[0]
                viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
                input_cond2 = (viewdirs[idx].unsqueeze(-1) * viewfreq).flatten(-2)
                input_cond2 = torch.cat([viewdirs[idx], input_cond2.sin(), input_cond2.cos()], -1)
                input_cond2 = input_cond2.reshape(1, HW[0], HW[1], -1).movedim(-1, 1).detach()
                input_cond = torch.cat((input_cond1, input_cond2), dim=1)

            if args.test_tile:
                rgb_srtest = net_sr.tile_process(rgbtest, input_cond, tile_size=args.test_tile)
            else:
                rgb_srtest = net_sr(rgbtest, input_cond).detach().to('cpu')

            rgb_srsave = rgb_srtest.squeeze().movedim(0, -1).detach().clamp(0, 1).numpy()
            rgbsr.append(rgb_srsave)
        rgbsr = np.array(rgbsr)
        for i in trange(len(rgbsr)):
            rgb8 = utils.to8b(rgbsr[i])
            filename = os.path.join(testsavedir, '{}_{:03d}.png'.format(ckpt_name, i))
            imageio.imwrite(filename, rgb8)


    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, _, viewdirs, rgb_features = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        net_sr = sr_esrnet.SFTNet(n_in_colors=cfg.fine_model_and_render.dim_rend, scale=sr_ratio, num_feat=64, num_block=5, num_grow_ch=32, num_cond=cfg.fine_model_and_render.num_cond, dswise=False).to(device)
        if args.sr_path:
            net_sr.load_network(load_path=args.sr_path, device=device)
        else:
            sr_load_path = os.path.join(cfg.basedir, cfg.expname, 'render_val/sresrnet_latest.pth')
            net_sr.load_network(load_path=sr_load_path, device=device)
        net_sr.eval()
        rgbsr = []
        for idx, rgbsave in enumerate(tqdm(rgb_features)):
            rgbtest = torch.from_numpy(rgbsave).movedim(-1, 0).unsqueeze(0).to(device)
            rgb = torch.from_numpy(rgbs[idx]).movedim(-1, 0).unsqueeze(0).to(device)
            
            if cfg.fine_model_and_render.num_cond == 1:
                input_cond = torch.from_numpy(depths).movedim(-1, 1)
                input_cond = input_cond[idx, :, :, :].to(device)
            elif cfg.fine_model_and_render.num_cond == 63:
                HW = data_dict['HW'][data_dict['i_val']].astype(int)[0]
                viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
                input_cond = (viewdirs[idx].unsqueeze(-1) * viewfreq).flatten(-2)
                input_cond = torch.cat([viewdirs[idx], input_cond.sin(), input_cond.cos()], -1)
                input_cond = input_cond.reshape(1, HW[0], HW[1], -1).movedim(-1, 1).detach()
            elif cfg.fine_model_and_render.num_cond == 64:
                input_cond1 = torch.from_numpy(depths[idx]).movedim(-1, 0)
                input_cond1 = input_cond1[:, idx:idx+1, :, :].to(device)
                HW = data_dict['HW'][data_dict['i_val']].astype(int)[0]
                viewfreq = torch.FloatTensor([(2**i) for i in range(10)]).to(device)
                input_cond2 = (viewdirs[idx].unsqueeze(-1) * viewfreq).flatten(-2)
                input_cond2 = torch.cat([viewdirs[idx], input_cond2.sin(), input_cond2.cos()], -1)
                input_cond2 = input_cond2.reshape(1, HW[0], HW[1], -1).movedim(-1, 1).detach()
                input_cond = torch.cat((input_cond1, input_cond2), dim=1)

            torch.cuda.synchronize()
            time_srstart = time.time()
            if args.test_tile:
                rgb_srtest = net_sr.tile_process(rgbtest, input_cond, tile_size=args.test_tile)
            else:
                rgb_srtest = net_sr(rgbtest, input_cond).detach()
            torch.cuda.synchronize()
            print(f'sr time is: {time.time() - time_srstart}')
            rgb_srtest = rgb_srtest.to('cpu')
            
            rgb_srsave = rgb_srtest.squeeze().movedim(0, -1).detach().clamp(0, 1).numpy()
            rgbsr.append(rgb_srsave)
        rgbsr = np.array(rgbsr)

        imageio.mimwrite(os.path.join(testsavedir, f'{cfg.expname}_sr.mp4'), utils.to8b(rgbsr), fps=25, codec='libx264', quality=8)
        imageio.mimwrite(os.path.join(testsavedir, f'{cfg.expname}_dvgo.mp4'), utils.to8b(rgbs), fps=25, codec='libx264', quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=25, quality=8)

    print('Done')
