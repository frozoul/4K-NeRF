# 4K-NeRF: High Fidelity Neural Radiance Fields at Ultra High Resolutions
Zhongshu Wang, Lingzhi Li, Zhen Shen, Li Shen, Liefeng Bo

Alibaba Group

[[Project Webpage](https://frozoul.github.io/4knerf/)] [[Bibtex](bib.txt)]

### 4K-Results
Due to the limitation of video size displayed on the webpage, the following videos are down-sampled. For the original 4K video comparison, please check under the `4K_results` folder.

https://user-images.githubusercontent.com/15401551/206893466-7bb285f7-67e4-42b5-9fa9-d3e9e784c197.mp4

https://user-images.githubusercontent.com/15401551/206893643-a4c09b2b-fb7b-4af9-aa4c-8acf006f4e07.mp4

## Setup
### Dependencies

```sh
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.


### Datasets

[nerf_llff_data](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) is the mainly used dataset as it is the only dataset that has 4K resolution images. 

[nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) was used in some ablation studys.

Put them in the `./datasets` sub-folder.

### Pre Trained Model
 We partially initialize VC-Decoder with a pretrained  [model](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)  to  speedup convergence. Put the downloaded pretrained weight in `./pretrained` sub-folder. Note that 4K-NeRF can be trained without pretrained weights.

### Directory-structure:
```
4K-NeRF
│ 
│
├──pretrained
│   └──RealESRNet_x4plus.pth
│ 
│ 
└── datasets
    ├── nerf_synthetic     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [chair|drums|ficus|hotdog|lego|materials|mic|ship]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       └── transforms_[train|val|test].json
    │
    │
    └── nerf_llff_data     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
        └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
```


## Training
Our method can train from scratch for any given scene, but we recommend pre-train the VC-Encoder for faster convergence:

LLFF: ` python run.py --config configs/llff/fern_lg_pretrain.py --render_test `

NeRF_Synthetic: ` python run.py --config configs/syn/syn_default.py --render_test `


After the pre-training, we use following commands to train the full 4K-NeRF with different configs:

* traing 4K resolution with L1 loss:

    ```
    python run_sr.py --config configs/llff/fern_lg_joint_l1.py \
           --render_test --ftdv_path logs/llff/pretrain_fern_l1/fine_last.tar \
           --ftsr_path ./pretrained/RealESRNet_x4plus.pth --test_tile 510
    ```

* traing 4K resolution with L1+GAN loss:

    ```
    python run_sr.py --config configs/llff/fern_lg_joint_l1+gan.py \
            --render_test --ftdv_path logs/llff/pretrain_fern_l1/fine_last.tar \
            --ftsr_path ./pretrained/RealESRNet_x4plus.pth --test_tile 510
    ```

* traing 1K resolution (LLFF) with L1+GAN loss:

    ```
    python run_sr.py --config configs/llff/1x_fern_lg_joint_l1+gan.py \
            --render_test --ftdv_path logs/llff/pretrain_fern_l1/fine_last.tar \
            --ftsr_path ./pretrained/RealESRNet_x4plus.pth 
    ```

* traing 1K resolution (NeRF_Synthetic) with L1+GAN loss:

    ```
    python run_sr.py --config configs/syn/1x_chair_joint_l1+gan.py \
             --render_test --ftdvcoa_path ./logs/syn/pretrain_chair/coarse_last.tar \
             --ftdv_path ./logs/syn/pretrain_chair/fine_last.tar \
             --ftsr_path ./pretrained/RealESRNet_x4plus.pth 
    ```

## Evaluation

Evaluate at 4K resolution:

   ```
   python run_sr.py --config configs/fern_lg_joint_l1+gan.py \
           --render_test --render_only --dv_path logs/llff/<eval_dir>/render_val/lpips_dvgo.tar \
           --sr_path logs/llff/<eval_dir>/render_val/sresrnet_latest.pth --test_tile 510
   ```

 Replace the `<eval_dir>` to the corresponding experiment name.

# Reference
* [Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction](https://github.com/sunset1995/DirectVoxGO)
* [BasicSR: Open Source Image and Video Restoration Toolbox](https://github.com/XPixelGroup/BasicSR)
