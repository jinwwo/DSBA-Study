
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Conv_Blocks import Inception_Block_V1
from layers.crackformer_case1_patching_EncDec import Encoder, EncoderLayer, FullAttention, AttentionLayer

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class UpscaleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpscaleConv, self).__init__()
        self.up_scaling = nn.ConvTranspose2d(in_channel, 
                                            out_channel, 
                                            kernel_size = 3, 
                                            stride = 2, 
                                            padding = 1)
    def forward(self, inputs):
        up_scaled = self.up_scaling(inputs)
        return up_scaled

class DownscaleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownscaleConv, self).__init__()
        self.Down_scaling = nn.Conv2d(in_channel, 
                                    out_channel, 
                                    kernel_size = 3, 
                                    stride = 2, 
                                    padding = 1)
    def forward(self, inputs):
        down_scaled = self.Down_scaling(inputs)
        return down_scaled

class MGRL_block(nn.Module):
    def __init__(self, configs):
        super(MGRL_block, self).__init__()

        # self.up_scale1 = UpscaleConv(16, 32)
        # self.up_scale2 = UpscaleConv(32, 64) 
        # self.up_scale3 = UpscaleConv(64, 128)
        # self.up_scale4 = UpscaleConv(128, 256)
        
        # self.down_scale4 = DownscaleConv(256, 128)
        # self.down_scale3 = DownscaleConv(128, 64)
        # self.down_scale2 = DownscaleConv(64, 32)
        # self.down_scale1 = DownscaleConv(32, 16)

        self.up_scale1 = UpscaleConv(configs.d_model, configs.d_model)
        self.up_scale2 = UpscaleConv(configs.d_model, configs.d_model) 
        self.up_scale3 = UpscaleConv(configs.d_model, configs.d_model)
        # self.up_scale4 = UpscaleConv(configs.d_model, configs.d_model)
        
        # self.down_scale4 = DownscaleConv(configs.d_model, configs.d_model)
        self.down_scale3 = DownscaleConv(configs.d_model, configs.d_model)
        self.down_scale2 = DownscaleConv(configs.d_model, configs.d_model)
        self.down_scale1 = DownscaleConv(configs.d_model, configs.d_model)
        
        self.fine_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                                        configs.d_model, 
                                        configs.n_heads,
                                        configs.d_projection,
                                        configs.patch_size),
                    configs.d_model,
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model), configs=configs
        )

        self.couarse_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), 
                                        configs.d_model, 
                                        configs.n_heads,
                                        configs.d_projection,
                                        configs.patch_size),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model), configs=configs
        )

        self.final_layer = nn.Sequential(nn.Conv2d(configs.d_model, configs.d_model, 1, padding=0),
                                    nn.Dropout(configs.dropout))
        
        # self.coarse_grained = series_decomp(5)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        # res, coarse = self.coarse_grained(input.reshape(B, C, H*W)s)
        # input = res.reshape(B, C, H, W)
        
        # coarse_grained_rep, attns = self.couarse_encoder(coarse.reshape(B, C, H, W))

        # upcaling part
        out_1 = self.up_scale1(inputs)
        out_2 = self.up_scale2(out_1)
        out_3 = self.up_scale3(out_2)
        # out_4 = self.up_scale4(out_3)
        
        # downscaling part
        # dec_out_4 = self.down_scale4(out_4)
        dec_out_3 = self.down_scale3(out_3)
        dec_out_2 = self.down_scale2(dec_out_3)
        dec_out_1 = self.down_scale1(dec_out_2)

        rep1, attns = self.fine_encoder(dec_out_1)
        
        # output = self.final_layer((rep1 + rep2)/2)
        output = self.final_layer(rep1)
        return output