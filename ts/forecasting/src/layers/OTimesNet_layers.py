import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.double_conv(x)

class OTimesNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OTimesNetBlock, self).__init__()
        
        self.enc_conv1 = DoubleConv(in_channels, out_channels)
        self.enc_conv2 = DoubleConv(in_channels, out_channels)
        self.enc_conv3 = DoubleConv(in_channels, out_channels)
        self.enc_conv4 = DoubleConv(in_channels, out_channels)
        self.enc_conv5 = DoubleConv(in_channels, out_channels)
        
        # up_scaling
        self.up_scaling1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        self.up_scaling2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)

        self.up_scaling_back1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        self.up_scaling_back2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)

        # down_scaling
        self.down_scaling1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.down_scaling2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

        self.down_scaling_back1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.down_scaling_back2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)

        self.dec_conv1 = DoubleConv(in_channels * 2, out_channels)
        self.dec_conv2 = DoubleConv(in_channels * 2, out_channels)
        self.dec_conv3 = DoubleConv(in_channels * 2, out_channels)
        
    def mask_specific_size(self, input_tensor, mask_size = (3,3), num_masked = 5):
        batch_size, num_channels, height, width = input_tensor.size()
        masksize_H, masksize_W = mask_size
        if height < masksize_H or width < masksize_W:
            return input_tensor 

        num_regions_H = height - masksize_H + 1
        num_regions_W = width - masksize_W + 1
        total_regions = num_regions_H * num_regions_W

        num_masked_regions = num_masked

        random_indices = torch.randperm(total_regions)[:num_masked_regions]

        start_h = random_indices // num_regions_W
        start_w = random_indices % num_regions_W

        mask = torch.ones_like(input_tensor)

        for i in range(num_masked_regions):
            h = start_h[i]
            w = start_w[i]
            mask[:, :, h:h + masksize_H, w:w + masksize_W] = 0
            
        x_masked = input_tensor * mask
        return x_masked
        
    def forward(self, x):
        masked_x = self.mask_specific_size(x)
        orginal_conv = self.enc_conv1(masked_x)
        
        up1 = self.up_scaling1(orginal_conv)
        down1 = self.down_scaling1(orginal_conv)
        
        up1_conv = self.enc_conv2(up1)
        down1_conv = self.enc_conv3(down1)
        
        up2 = self.up_scaling2(up1_conv)
        down2 = self.down_scaling2(down1_conv)
        
        up2_conv = self.enc_conv4(up2)
        down2_conv = self.enc_conv5(down2)
        
        up1_back = self.up_scaling_back1(up2_conv)
        down1_back = self.down_scaling_back1(down2_conv)
        
        up1_back_conv = self.dec_conv1(torch.cat([up1_back, up1_conv], dim=1))
        down1_back_conv = self.dec_conv2(torch.cat([down1_back, down1_conv], dim=1))
        
        up_back = self.up_scaling_back2(up1_back_conv)
        down_back = self.down_scaling_back2(down1_back_conv)
        
        combined = torch.stack([up_back, down_back], dim=0)
        mean_tensor = torch.mean(combined, dim=0)
        
        final_result = self.dec_conv3(torch.cat([mean_tensor, orginal_conv], dim=1))
        
        return final_result