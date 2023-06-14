import copy
import math
import numpy as np

from os.path import join as pjoin

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from . import trans_configs as configs
from .trans_modeling_resnet_skip import ResNetV2
import torch.nn.functional as F

from .HGFilters import *
from ..net_util import init_net
import torchvision.transforms as transforms

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        config.transformer["num_heads"] = 16
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, opt, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        # in_channels = 8
        # patch_size = 32
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        # self.patch_embeddings_3D = nn.Conv3d(in_channels=32,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=3)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # self.position_embeddings = nn.Parameter(torch.zeros(1, 5000, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 128, 512))

        self.dropout = Dropout(config.transformer["dropout_rate"])

        # self.vrn_filter = VrnNet(opt)
        c_len_deepvoxels = 32
        self.trans_Unet = TransUnet3D(c_len_in=c_len_deepvoxels, c_len_out=c_len_deepvoxels)
        # self.brightness_change = transforms.ColorJitter(contrast=(0.1, 0.9))


    def forward(self, image):
        # patch_size (2,1,1)
        # n_patches (8/2)*(8/1)*(8/1) = 4*8*8 = 256
        # hidden = 2*1*1*128 = 256
        # image = self.brightness_change(image)
        if self.hybrid:
            x, features_2D = self.hybrid_model(image)
        else:
            x = image
            features_2D = None
        # print(x1.shape)

        # x = self.patch_embeddings_3D(x)
        # x = x.flatten(2)
        # x = x.flatten(2)

        x = x.view(x.shape[0], -1, x.shape[-1], x.shape[-2], x.shape[-1])  # (B,1024,32,32) -> (B,32,32,32,32)
        x, features_3D, indice = self.trans_Unet.encoder(x) # (B,256,4,8,8)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)


        # x = x.view(x.shape[0], -1, x.shape[-1], x.shape[-2], x.shape[-1])  # (B,8,32,48,32)
        # x = x.transpose(2, 1)  # (B,32,8,48,32)
        # x = x.transpose(1, 0)  # (32,B,8,48,32)
        # embeddings = []
        # for n in range(x.shape[0]):
        #     x_ = self.patch_embeddings(x[n])
        #     x_ = x_.squeeze(dim=3)
        #     x_ = x_.squeeze(dim=2)
        #     x_ = x_.unsqueeze(dim=0)
        #     embeddings.append(x_)
        # x = torch.cat(embeddings,dim=0)
        # # x = torch.from_numpy(x)
        # x = x.transpose(1, 0)  # (B,32,768)

        # x = self.patch_embeddings(x1)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features_2D, features_3D, indice


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        config.transformer["num_layers"] = 24 #12
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            1024,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        c_len_deepvoxels = 32
        self.trans_Unet = TransUnet3D(c_len_in=c_len_deepvoxels,c_len_out=c_len_deepvoxels)

    def forward(self, hidden_states, indice, features_2D=None, features_3D=None):
        B, n_patch, hidden = hidden_states.size()   # (B, 256, 256)
        d, h, w = 4, 4, 8
        x = hidden_states.contiguous().view(B, -1, h, w, hidden)    # (B, 4, 8, 8, 256)
        x = x.transpose(4, 1)   # (B, 256, 8, 8, 4)
        x = x.transpose(4, 2)   # (B, 256, 4, 8, 8)
        x = x.transpose(4, 3)   # (B, 256, 4, 8, 8)
        x = self.trans_Unet.decoder(x, indice, features_3D) # (B, 32, 32, 32, 32)
        x = x.view(B, -1, 32, 32)   # (B, 1024, 32, 32)
        # print("111",x.shape)

        # B, n_patch, hidden = hidden_states.size()
        # h, w = 48, 32
        # x = hidden_states.transpose(1, 0)   #(n_patch, B, hidden)
        # x = x.unsqueeze(dim=3)
        # x = x.unsqueeze(dim=4)   #(n_patch, B, hidden, 1, 1)
        # embeddings = []
        # for n in range(x.shape[0]):
        #     x_ = self.conv_1(x[n])
        #     x_ = F.interpolate(x_, scale_factor=16, mode='bilinear')
        #     x_ = x_.unsqueeze(dim=0)
        #     embeddings.append(x_)
        # x = torch.cat(embeddings, dim=0)
        # x = x.transpose(1, 0)  # (B,32,8,32,32)
        # x = x.transpose(2, 1)  # (B,8,32,32,32)


        # x = x.contiguous().view(B, -1, h, w)
        # # print(x.shape)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)   # (B, 512, 32, 32)
        for i, decoder_block in enumerate(self.blocks):
            # if i == 1:
            #     x = F.interpolate(x, scale_factor=2, mode='bilinear')
            #     break
            if features_2D is not None:
                skip = features_2D[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)     # i=0 (B, 256, 64, 64) ; i=1 (B, 128, 128, 128); i=2 (B, 64, 256, 256); i=3 (B, 16, 512, 512)
        return x

class TransUnet3D(nn.Module):

    def __init__(self, c_len_in, c_len_out, opt=None):

        super(TransUnet3D, self).__init__()

        # (B,32,32,32,32) | conv3d(k3,s1,i32,o32,nb), BN3d, LearkyReLU(0.2) | (B,32,32,32,32) ------> skip-0: (B,32,32,32,32)
        c_len_1 = 32
        self.conv3d_pre_process = nn.Sequential(Conv3dSame(c_len_in, c_len_1, kernel_size=3, bias=False), nn.BatchNorm3d(c_len_1, affine=True), nn.LeakyReLU(0.2, True))

        # (B,32,32,32,32) | conv3d(k4,s2,i32,o64,nb), BN3d, LeakyReLU(0.2) | (B,64,16,16,16) ------> skip-1: (B,64,16,16,16)
        c_len_2 = 64
        self.conv3d_enc_1 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c_len_1, c_len_2, kernel_size=4, padding=0, stride=2, bias=False), nn.BatchNorm3d(c_len_2, affine=True), nn.LeakyReLU(0.2, True))

        # (B,64,16,16,16) | conv3d(k4,s2,i64,o128,nb), LeakyReLU(0.2) | (B,128,8,8,8)
        c_len_3 = 128
        self.conv3d_enc_2 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c_len_2, c_len_3, kernel_size=4, padding=0, stride=2, bias=False), nn.BatchNorm3d(c_len_3, affine=True), nn.LeakyReLU(0.2, True))

        # (B,128,8,8,8) | conv3d(k3,s1,i128,o256,b), LeakyReLU(0.2) | (B,256,8,8,8)
        c_len_4 = 512
        self.conv3d_embbeding_process = nn.Sequential(Conv3dSame(c_len_3, c_len_4, kernel_size=3, bias=True), nn.LeakyReLU(0.2, True))

        # (B,256,8,8,8) | maxpool3d(k(2,1,1),s(2,1,1),i256,o256,b) | (B,256,4,8,8)
        self.maxpool3d_enc_2 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), return_indices=True))

        # (B,256,4,8,8) | maxunpool3d(k(2,1,1),s(2,1,1),i256,o256,b) | (B,256,8,8,8)
        self.maxunpool3d_dec_2 = nn.MaxUnpool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        # (B,256,8,8,8) | DeConv3d(k3,s1,i256,o128,b), ReLU | (B,128,8,8,8)
        self.deconv3d_process = nn.Sequential(Conv3dSame(c_len_4, c_len_3, kernel_size=3, bias=True), nn.ReLU(True))

        # (B,128,8,8,8) | DeConv3d(k4,s2,i128,o64,nb), ReLU | (B,64,16,16,16)
        self.deconv3d_dec_2 = nn.Sequential(nn.ConvTranspose3d(c_len_3, c_len_2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm3d(c_len_2, affine=True), nn.ReLU(True))

        # (B,64+64,16,16,16) | DeConv3d(k4,s2,i128,o32,nb), BN3d, ReLU | (B,32,32,32,32) <------ skip-1: (B,32,32,32,32)
        self.deconv3d_dec_1 = nn.Sequential(nn.ConvTranspose3d(c_len_2 * 2, c_len_1, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm3d(c_len_1, affine=True), nn.ReLU(True))

        # (BV,32+32,32,32,32) | Conv3d(k3,s1,i64,o32,nb), BN3d, ReLU | (B,32,32,32,32) <------ skip-0: (B,32,32,32,32)
        self.conv3d_final_process = nn.Sequential(Conv3dSame(c_len_1 * 2, c_len_out, kernel_size=3, bias=False), nn.BatchNorm3d(c_len_out, affine=True), nn.ReLU(True))

    def encoder(self, x):
        """
        e.g. in-(B,32,32,32,32), out-(B,256,4,8,8)
        """
        skip_encoder_list = []

        # (B,32,32,32,32) | conv3d(k3,s1,i32,o32,nb), BN3d, LearkyReLU(0.2) | (B,32,32,32,32) ------> skip-0: (B,32,32,32,32)
        x = self.conv3d_pre_process(x)
        skip_encoder_list.append(x)

        # (B,32,32,32,32) | conv3d(k4,s2,i32,o64,nb), BN3d, LeakyReLU(0.2) | (B,64,16,16,16) ------> skip-1: (B,64,16,16,16)
        x = self.conv3d_enc_1(x)
        skip_encoder_list.append(x)

        # (B,64,16,16,16) | conv3d(k4,s2,i64,o128,nb), LeakyReLU(0.2) | (B,128,8,8,8)
        x = self.conv3d_enc_2(x)

        # (B,128,8,8,8) | conv3d(k3,s1,i128,o256,b), LeakyReLU(0.2) | (B,256,8,8,8)
        x = self.conv3d_embbeding_process(x)

        # (B,256,8,8,8) | maxpool3d(k(2,1,1),s(2,1,1),i256,o256,b) | (B,256,4,8,8)
        x, indice = self.maxpool3d_enc_2(x)
        # indice_list = []
        # tmp = []
        # for i in range(x.shape[0]):
        #     print(x[i].shape)
        #     x0, indice = self.avgpool3d_enc_2(x[i])
        #     x0 = x0.unsqueeze(0)
        #     tmp.append(x0)
        #     indice_list.append(indice)
        # x = torch.cat(tmp,dim=0)
        return x, skip_encoder_list, indice

    def decoder(self, x, indice, skip_encoder_list=None):
        """
        e.g. in-(B,256,4,8,8), out-(B,32,32,32,32)
        """
        # x = self.deconv3d_dec_3(x)
        # print(x.shape)
        # # (BV,32,8,12,8) | DeConv3d(k4,s2,i32,o16,b), ReLU | (BV,16,16,24,16)
        # x = torch.cat([skip_encoder_list[2], x], dim=1)
        # tmp = []
        # for i in range(x.shape[0]):
        #     x0 = self.avgunpool3d_dec_2(x[i], indice[i])
        #     x0 = x0.unsqueeze(0)
        #     tmp.append(x0)
        # x = torch.cat(tmp,dim=0)

        # (B,256,4,8,8) | maxunpool3d(k(2,1,1),s(2,1,1),i256,o256,b) | (B,256,8,8,8)
        x = self.maxunpool3d_dec_2(x, indice)

        # (B,256,8,8,8) | DeConv3d(k3,s1,i256,o128,b), ReLU | (B,128,8,8,8)
        x = self.deconv3d_process(x)

        # (B,128,8,8,8) | DeConv3d(k4,s2,i128,o64,nb), ReLU | (B,64,16,16,16)
        x = self.deconv3d_dec_2(x)

        # (B,64+64,16,16,16) | DeConv3d(k4,s2,i128,o32,nb), BN3d, ReLU | (B,32,32,32,32) <------ skip-1: (B,32,32,32,32)
        x = torch.cat([skip_encoder_list[1], x], dim=1)
        x = self.deconv3d_dec_1(x)

        # (BV,32+32,32,32,32) | Conv3d(k3,s1,i64,o32,nb), BN3d, ReLU | (B,32,32,32,32) <------ skip-0: (B,32,32,32,32)
        x = torch.cat([skip_encoder_list[0], x], dim=1)
        x = self.conv3d_final_process(x)

        return x

# class VrnNet(nn.Module):
#     def __init__(self, opt, projection_mode='orthogonal'):
#         super(VrnNet, self).__init__()
#
#         # ----- init. -----
#
#         self.name = 'vrn'
#         self.opt = opt
#         self.hourglass_dim = 1024
#         self.im_feat_list = []  # a list of deep voxel features
#         self.intermediate_preds_list = []  # a list of estimated occupancy grids
#         self.intermediate_3d_gan_pred_fake_gen = []  # a list of 3d gan discriminator output on fake est., for training generator
#         self.intermediate_3d_gan_pred_fake_dis = []  # a list of 3d gan discriminator output on fake est., for training discriminator
#         self.intermediate_3d_gan_pred_real_dis = None  # a BATCH of 3d gan discriminator output on real gt., for training discriminator
#         self.intermediate_render_list = []  # a list of rendered rgb images
#         self.intermediate_pseudo_inverseDepth_list = []  # a list of pseudo inverse-depth maps
#         self.intermediate_render_discriminator_list = []  # a list of patch-GAN discriminator values of the rendered rgb images
#
#         # ----- generate deep voxels -----
#         if True:
#
#             # (BV,3,512,512) | resize                                       | (BV,3,384,384)
#             # (BV,3,384,384) | crop                                         | (BV,3,384,256)
#             # (BV,3,384,256) | conv2d(k7,s2,p3,i3,o64,b), GN(g32,c64), ReLU | (BV,64,192,128)
#             # c_len_1 = 64
#             # self.conv1 = nn.Sequential(nn.Conv2d(3, c_len_1, kernel_size=7, stride=2, padding=3),
#             #                            nn.GroupNorm(32, c_len_1), nn.ReLU(True))
#             #
#             # # (BV,64,192,128)  | residual_block(i64,o128,GN) | (BV,128,192,128)
#             # # (BV,128,192,128) | avg_pool2d(k2,s2)           | (BV,128,96,64)
#             # c_len_2 = 128
#             # self.conv2 = ConvBlock(c_len_1, c_len_2, self.opt.norm)
#             #
#             # # (BV,128,96,64) | residual_block(i128,o128,GN) | (BV,128,96,64)
#             # # (BV,128,96,64) | avg_pool2d(k2,s2)            | (BV,128,48,32)
#             # c_len_3 = 128
#             # self.conv3 = ConvBlock(c_len_2, c_len_3, self.opt.norm)
#             #
#             # # (BV,128,48,32) | residual_block(i128,o128,GN) | (BV,128,48,32)
#             # # (BV,128,48,32) | residual_block(i128,o256,GN) | (BV,256,48,32)
#             # c_len_4 = 128
#             # self.conv4 = ConvBlock(c_len_3, c_len_4, self.opt.norm)
#             # c_len_5 = 256
#             # self.conv5 = ConvBlock(c_len_4, c_len_5, self.opt.norm)
#             # c_len_6 = 512
#             # self.conv6 = ConvBlock(c_len_5, c_len_6, self.opt.norm)
#             # c_len_7 = 768
#             # self.conv7 = ConvBlock(c_len_6, c_len_7, self.opt.norm)
#
#             # (BV,256,48,32) | 4-stack-hour-glass | BCDHW of (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)
#             c_len_deepvoxels = 32
#             for hg_module in range(self.opt.vrn_num_modules):  # default: 4
#
#                 self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 1024, self.opt.norm, self.opt.upsample_mode))
#
#                 self.add_module('top_m_' + str(hg_module), ConvBlock(1024, 1024, self.opt.norm))
#                 self.add_module('conv_last' + str(hg_module), nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
#                 if self.opt.norm == 'batch':
#                     self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(1024))
#                 elif self.opt.norm == 'group':
#                     self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 1024))
#
#                 self.add_module("branch_out_3d_trans_unet" + str(hg_module), TransUnet3D(c_len_in=c_len_deepvoxels, c_len_out=c_len_deepvoxels))
#
#                 self.add_module('l' + str(hg_module), nn.Conv2d(1024, self.hourglass_dim, kernel_size=1, stride=1, padding=0))
#
#                 if hg_module < self.opt.vrn_num_modules - 1:
#                     self.add_module('bl' + str(hg_module), nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
#                     self.add_module('al' + str(hg_module), nn.Conv2d(self.hourglass_dim, 1024, kernel_size=1, stride=1, padding=0))
#
#     def forward(self, images):
#         # (BV,3,512,512) | resize                                       | (BV,3,384,384)
#         # (BV,3,384,384) | crop                                         | (BV,3,384,256)
#         # (BV,3,384,256) | conv2d(k7,s2,p3,i3,o64,b), GN(g32,c64), ReLU | (BV,64,192,128)
#         # images = F.interpolate(images, size=self.opt.vrn_net_input_height, mode='bilinear', align_corners=True)
#         # images = images[:, :, images.shape[-2] // 2 - self.opt.vrn_net_input_width // 2:images.shape[
#         #                                                                                        -2] // 2 + self.opt.vrn_net_input_width // 2, images.shape[-1] // 2 - self.opt.vrn_net_input_width // 2:images.shape[
#         #                                                                                        -1] // 2 + self.opt.vrn_net_input_width // 2]
#         # images = self.conv1(images)
#         #
#         # # (BV,64,192,128)  | residual_block(i64,o128,GN) | (BV,128,192,128)
#         # # (BV,128,192,128) | avg_pool2d(k2,s2)           | (BV,128,96,64)
#         # images = self.conv2(images)
#         # images = F.avg_pool2d(images, 2, stride=2)
#         #
#         # # (BV,128,96,64) | residual_block(i128,o128,GN) | (BV,128,96,64)
#         # # (BV,128,96,64) | avg_pool2d(k2,s2)            | (BV,128,48,32)
#         # images = self.conv3(images)
#         # images = F.avg_pool2d(images, 2, stride=2)
#         #
#         # # (BV,128,48,32) | residual_block(i128,o128,GN) | (BV,128,48,32)
#         # # (BV,128,48,32) | residual_block(i128,o256,GN) | (BV,256,48,32)
#         # images = self.conv4(images)
#         # images = self.conv5(images)
#
#         # (BV,256,48,32) | 4-stack-hour-glass | BCDHW of (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)
#         previous = images
#         self.im_feat_list = []
#         self.skip_encoder_list = []
#         self.indice_list = []
#         for i in range(self.opt.vrn_num_modules):  # default: 4
#
#             hg = self._modules['m' + str(i)](previous)
#
#             ll = hg
#             ll = self._modules['top_m_' + str(i)](ll)
#
#             ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)
#
#             # Predict heatmaps
#             tmp_out = self._modules['l' + str(i)](ll)  # (B,1024,32,32)
#             assert (tmp_out.shape[1] % tmp_out.shape[-1] == 0)
#             tmp = tmp_out.view(tmp_out.shape[0], -1, tmp_out.shape[-1], tmp_out.shape[-2], tmp_out.shape[-1])  # (B,32,32,32,32)
#             tmp, skip_encoder, indice = self._modules['branch_out_3d_trans_unet' + str(i)].encoder(tmp)  # (B,256,4,8,8)
#             # feature = self._modules['branch_out_3d_trans_unet' + str(i)].decoder(tmp,skip_encoder)  # (BV,8,32,48,32)
#             # if self.training:
#
#             self.im_feat_list.append(tmp)
#             self.skip_encoder_list.append(skip_encoder)
#             self.indice_list.append(indice)
#             # else:
#             #
#             #     if i == (self.opt.vrn_num_modules - 1): self.im_feat_list.append(tmp_out)
#             # tmp_out = tmp_out.view(tmp_out.shape[0], -1, tmp_out.shape[-2], tmp_out.shape[-1])  # (BV,256,48,32)
#
#             if i < (self.opt.vrn_num_modules - 1):
#                 ll = self._modules['bl' + str(i)](ll)
#                 tmp_out_ = self._modules['al' + str(i)](tmp_out)
#                 previous = previous + ll + tmp_out_
#         return self.im_feat_list[-1], self.skip_encoder_list[-1], self.indice_list[-1]

class HGFilter(nn.Module):
    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack  # default: 4

        self.opt = opt

        # Base part
        # if opt.normal_activate:
        #     self.conv1 = nn.Conv2d(19, 64, kernel_size=7, stride=2, padding=3)
        # else:
        #     self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3)
        # if opt.depth_activate:
        self.conv1 = nn.Conv2d(22, 64, kernel_size=7, stride=2, padding=3)

        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt.norm)
        self.conv4 = ConvBlock(128, 256, self.opt.norm)

        # Stacking part
        for hg_module in range(self.num_modules):  # default: 4
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt.norm, self.opt.upsample_mode))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt.norm))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))

            self.add_module('l' + str(hg_module), nn.Conv2d(256, opt.hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

            if hg_module == (self.num_modules - 1) and self.opt.recover_dim:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

        # recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512
        if self.opt.recover_dim:
            self.recover_dim_match_fea_1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
            self.recover_dim_conv_1 = ConvBlock(256, 256, self.opt.norm)
            self.recover_dim_match_fea_2 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0)
            self.recover_dim_conv_2 = ConvBlock(256, 256, self.opt.norm)

    def forward(self, x):
        '''
        Filter the input images, store all intermediate features.

        Input
            x: [B * num_views, C, H, W] input images, float -1 ~ 1, RGB

        Output
            outputs:       [(B * num_views, opt.hourglass_dim, H/4, W/4), (same_size), (same_size), (same_size)], list length is opt.num_stack
            tmpx.detach():  (B * num_views, 64, H/2, W/2)
            normx:          (B * num_views, 128, H/4, W/4)
        '''
        raw_x = x
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        outputs = []
        for i in range(self.num_modules):  # default: 4

            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            # recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512
            if i == (self.num_modules - 1) and self.opt.recover_dim:
                # merge features
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                fea_upsampled = previous + ll + tmp_out_  # (BV,256,128,128)

                # upsampling: (BV,256,128,128) to (BV,256,256,256)
                if self.opt.upsample_mode == "bicubic":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='bicubic',
                                                  align_corners=True)  # (BV,256,256,256)
                elif self.opt.upsample_mode == "nearest":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='nearest')  # (BV,256,256,256)
                else:

                    print("Error: undefined self.upsample_mode {} when self.opt.recover_dim {}!".format(
                        self.opt.upsample_mode, self.opt.recover_dim))
                fea_upsampled = fea_upsampled + self.recover_dim_match_fea_1(tmpx)
                fea_upsampled = self.recover_dim_conv_1(fea_upsampled)  # (BV,256,256,256)

                # upsampling: (BV,256,256,256) to (BV,256,512,512)
                if self.opt.upsample_mode == "bicubic":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='bicubic',
                                                  align_corners=True)  # (BV,256,512,512)
                elif self.opt.upsample_mode == "nearest":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='nearest')  # (BV,256,512,512)
                else:

                    print("Error: undefined self.upsample_mode {} when self.opt.recover_dim {}!".format(
                        self.opt.upsample_mode, self.opt.recover_dim))
                fea_upsampled = fea_upsampled + self.recover_dim_match_fea_2(raw_x)
                fea_upsampled = self.recover_dim_conv_2(fea_upsampled)  # (BV,256,512,512)

                outputs.append(fea_upsampled)

        return outputs, tmpx.detach(), normx

# class DPFilter(nn.Module):
#     def __init__(self, opt):
#         super(DPFilter, self).__init__()
#         self.num_modules = opt.num_stack  # default: 4
#
#         self.opt = opt
#
#         # Base part
#         if opt.normal_activate:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         else:
#             self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3)
#
#         if self.opt.norm == 'batch':
#             self.bn1 = nn.BatchNorm2d(64)
#         elif self.opt.norm == 'group':
#             self.bn1 = nn.GroupNorm(32, 64)
#
#         if self.opt.hg_down == 'conv64':
#             self.conv2 = ConvBlock(64, 64, self.opt.norm)
#             self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         elif self.opt.hg_down == 'conv128':
#             self.conv2 = ConvBlock(64, 128, self.opt.norm)
#             self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
#         elif self.opt.hg_down == 'ave_pool':
#             self.conv2 = ConvBlock(64, 128, self.opt.norm)
#         else:
#             raise NameError('Unknown Fan Filter setting!')
#
#         self.conv3 = ConvBlock(128, 128, self.opt.norm)
#         self.conv4 = ConvBlock(128, 256, self.opt.norm)
#
#     def forward(self, x):
#         '''
#         Filter the input images, store all intermediate features.
#
#         Input
#             x: [B * num_views, C, H, W] input images, float -1 ~ 1, RGB
#
#         Output
#             outputs:       [(B * num_views, opt.hourglass_dim, H/4, W/4), (same_size), (same_size), (same_size)], list length is opt.num_stack
#             tmpx.detach():  (B * num_views, 64, H/2, W/2)
#             normx:          (B * num_views, 128, H/4, W/4)
#         '''
#         raw_x = x
#         x = F.relu(self.bn1(self.conv1(x)), True)
#         tmpx = x
#         if self.opt.hg_down == 'ave_pool':
#             x = F.avg_pool2d(self.conv2(x), 2, stride=2)
#         elif self.opt.hg_down in ['conv64', 'conv128']:
#             x = self.conv2(x)
#             x = self.down_conv2(x)
#         else:
#             raise NameError('Unknown Fan Filter setting!')
#
#         normx = x
#
#         x = self.conv3(x)
#         x = self.conv4(x)
#
#         previous = x
#         outputs = previous
#
#         return outputs, tmpx.detach(), normx


class Transformer(nn.Module):
    def __init__(self, opt, config, img_size, vis):
        super(Transformer, self).__init__()
        config.hidden_size = 512
        self.embeddings = Embeddings(opt, config, img_size=img_size)
        self.encoder = Encoder(config, vis)
        self.decoder = DecoderCup(config)
        self.image_filter = HGFilter(opt)
        # self.depth_filter = DPFilter(opt)

    def forward(self, images, normal, depth=None):
        embedding_output, features_2D, features_3D, indice = self.embeddings(images)    # (B, 3, 512, 512)
        encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden) # (B, 256, 256)
        decoded = self.decoder(encoded, indice, features_2D, features_3D)
        if normal != None:
            new_feature = torch.cat([normal,decoded],dim=1)
        else:
            new_feature = decoded
        if depth != None:
            new_feature = torch.cat([depth,normal,decoded],dim=1)
        filter, _, _ = self.image_filter(new_feature)
        return filter

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}