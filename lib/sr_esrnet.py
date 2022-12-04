from copy import deepcopy
import torch
from torch import nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
import math
import os
import time


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class SFTLayer(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(num_grow_ch, num_grow_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(num_grow_ch, num_feat, 1)
        self.SFT_shift_conv0 = nn.Conv2d(num_grow_ch, num_grow_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(num_grow_ch, num_feat, 1)

    def forward(self, x, cond):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.2, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.2, inplace=True))
        return x * (scale + 1) + shift


class ResidualDenseBlock_SFT(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock_SFT, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.sft0 = SFTLayer(num_feat, num_grow_ch)
        self.sft1 = SFTLayer(num_grow_ch, num_grow_ch)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        xc0 = self.sft0(x[0], x[1])
        x1 = self.lrelu(self.conv1(xc0))
        x2 = self.lrelu(self.conv2(torch.cat((xc0, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((xc0, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((xc0, x1, x2, x3), 1)))
        xc1 = self.sft1(x4, x[1])
        x5 = self.conv5(torch.cat((xc0, x1, x2, x3, xc1), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return (x5 * 0.2 + x[0], x[1])


class RRDB_SFT(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB_SFT, self).__init__()
        self.rdb1 = ResidualDenseBlock_SFT(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_SFT(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_SFT(num_feat, num_grow_ch)
        self.sft0 = SFTLayer(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.sft0(out[0], x[1])
        # Emperically, we use 0.2 to scale the residual for better performance
        return (out * 0.2 + x[0], x[1])


class RRDBNet_bps(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, n_colors, scale, num_feat=64, num_block=5, num_grow_ch=32):
        super(RRDBNet_bps, self).__init__()
        self.scale = scale
        # if scale == 2:
        #     num_in_ch = num_in_ch * 4
        # elif scale == 1:
        #     num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(n_colors, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, n_colors, 3, 1, 1)

        self.ps_preconv1 = nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1)
        self.psup1 = nn.PixelShuffle(2)
        if self.scale == 4:
            self.ps_preconv2 = nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1)
            self.psup2 = nn.PixelShuffle(2)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # if self.scale == 2:
        #     feat = pixel_unshuffle(x, scale=2)
        # elif self.scale == 1:
        #     feat = pixel_unshuffle(x, scale=4)
        # else:
        #     feat = x
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(self.psup1(self.ps_preconv1(feat))))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(self.psup2(self.ps_preconv2(feat))))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def tile_process(self, img, tile_size, tile_pad=10):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape).to('cpu')
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                # try:
                #     with torch.no_grad():
                output_tile = self(input_tile)
                # except Exception as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile].detach().to('cpu')
        return output

    def load_network(self, load_path, device, strict=True, param_key='params_ema'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        # net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=device)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                print('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        print(f'Loading {self.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(load_net, strict)
        self.load_state_dict(load_net, strict=strict)

    def _print_different_keys_loading(self, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def save_network(self, save_root, net_label, current_iter, param_key='params'):

        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(save_root, save_filename)

        net = self if isinstance(self, list) else [self]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                print(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            print(f'Still cannot save {save_path}. Just ignore it.')


class SFTNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, n_in_colors, scale, num_feat=64, num_block=5, num_grow_ch=32, num_cond=1, dswise=False):
        super(SFTNet, self).__init__()
        self.scale = scale
        self.dswise = dswise
        # if scale == 2:
        #     num_in_ch = num_in_ch * 4
        # elif scale == 1:
        #     num_in_ch = num_in_ch * 16
        if dswise:
            self.conv_first = nn.Conv2d(n_in_colors, num_feat, 1)
        else:
            self.conv_first = nn.Conv2d(n_in_colors, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB_SFT, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if n_in_colors > 3:
            self.conv_fea = nn.Conv2d(n_in_colors, num_feat, 3, 1, 1)
            self.conv_prefea = nn.Conv2d(2*num_feat, num_feat, 3, 1, 1)

        # upsample
        if self.scale > 1:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.scale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sftbody = SFTLayer(num_feat, num_grow_ch)
        self.CondNet = nn.Sequential(
            nn.Conv2d(num_cond, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 1))

    def forward(self, x, cond, fea=None):
        if fea is None:
            feat = self.conv_first(x)
        else:
            feat_rgb = self.conv_first(x)
            # feat += torch.sigmoid(self.conv_fea(fea))
            feat = torch.cat((feat_rgb, fea), dim=1)
            feat = self.conv_prefea(feat)
        cond = self.CondNet(cond)
        body_feat = self.body((feat, cond))
        body_feat = self.sftbody(body_feat[0], body_feat[1])
        body_feat = self.conv_body(body_feat)
        body_feat += feat
        # upsample
        if self.scale > 1:
            body_feat = self.lrelu(self.conv_up1(F.interpolate(body_feat, scale_factor=2, mode='nearest')))
            if self.scale == 4:
                body_feat = self.lrelu(self.conv_up2(F.interpolate(body_feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(body_feat)))
        return out

    def tile_process(self, img, cond, tile_size, tile_pad=10):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape).to('cpu')
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                cond_tile = cond[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # upscale tile
                # try:
                #     with torch.no_grad():
                output_tile = self(input_tile, cond_tile)
                # except Exception as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile].detach().to('cpu')
        return output

    def load_network(self, load_path, device, strict=True, param_key='params_ema'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        # net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=device)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                print('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        print(f'Loading {self.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(load_net, strict)
        self.load_state_dict(load_net, strict=strict)

    def _print_different_keys_loading(self, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def save_network(self, save_root, net_label, current_iter, param_key='params'):

        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(save_root, save_filename)

        net = self if isinstance(self, list) else [self]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                print(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            print(f'Still cannot save {save_path}. Just ignore it.')