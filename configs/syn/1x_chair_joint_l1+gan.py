_base_ = '../default.py'

expname = 'sr_dvgo_chair_1x_gan'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='../datasets/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=True,
    factor=1,
    load_sr=1,
)

fine_train = dict(
    N_iters=300000,
    lrate_srnet=2e-4,
    weight_pcp=0.5,
    weight_gan=0.05,
    weight_style=0.2,
    ray_sampler='patch_inmask',
    N_patch=64,
    lrate_decay=300,
)

coarse_model_and_render = dict(
    dim_rend=3,
    act_type='relu',
)

fine_model_and_render = dict(
    mode_type='mlp',
    viewbase_pe=0,
    spatial_pe=0,
    num_cond=1,
    dim_rend=3,
    act_type='relu',
    d_model='Unet'
)
