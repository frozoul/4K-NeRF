_base_ = '../default.py'

expname = 'pretrain_chair'
basedir = './logs/syn'

data = dict(
    datadir='./datasets/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=True,
    load_sr=False,
)

coarse_model_and_render = dict(
    dim_rend=3,
    act_type='relu',
)
