_base_ = './llff_default_lg.py'

expname = 'pretrain_fern_l1'

data = dict(
    datadir='./datasets/nerf_llff_data/fern',
    dataset_type='llff',
    load_sr=0,
    width=None,
    height=None,
    factor=4,
    llffhold=8,
)

fine_train = dict(
    lrate_srnet=0,
    lrate_adainet=1e-3,
    weight_pcp=0,
    weight_gan=0
)

fine_model_and_render = dict(
    mode_type='mlp',
    viewbase_pe=0,
    spatial_pe=0,
    act_type='relu'
)
