_base_ = './llff_default_lg.py'

expname = 'joint_fern_l1+gan'

data = dict(
    datadir='./datasets/nerf_llff_data/fern',
    dataset_type='llff',
    load_sr=4,
    llffhold=8,
)

fine_train = dict(
    N_iters=300000,
    tv_dense_before=10000,
    lrate_srnet=2e-4,
    weight_pcp=0.5,
    weight_gan=0.05,
    weight_style=0.2,
    weight_entropy_last=0.001,
    tv_before=10000,
    ray_sampler='patch_mimg',
    N_patch=64,
    lrate_decay=300
)

fine_model_and_render = dict(
    mode_type='mlp',
    viewbase_pe=0,
    spatial_pe=0,
    num_cond=1,
    dim_rend=3,
    act_type='relu',
    d_model='Unet',
)
