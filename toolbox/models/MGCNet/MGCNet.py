import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.models.MGCNet.convnext_ori import convnext_small,LayerNorm

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class gate1(nn.Module):
    def __init__(self, cur_channel):
        super(gate1, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)




    def forward(self, x_cur,aux_1, aux_2, aux_3):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur_1)
        x_cur_3 = self.cur_b3(x_cur_2)
        x_cur_4 = self.cur_b4(x_cur_3)
        x_cur_all = self.cur_all(x_cur_4)
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = self.cur_all_sa(cur_all_ca)  #空间注意力权重

        # aux1 conv
        aux_1 = self.upsample2(aux_1)
        weight1 = self.sa1(aux_1)
        aux_1 = x_cur_all*weight1

        #aux_2 conv
        aux_2 = self.upsample4(aux_2)
        weight2 = self.sa2(aux_2)
        aux_2 = x_cur_all * weight2

        #aux_3 conv
        aux_3 = self.upsample8(aux_3)
        weight3 = self.sa3(aux_3)
        aux_3 = x_cur_all * weight3

        aux_fusion = aux_1 + aux_2 + aux_3

        weight = 1-cur_all_sa
        cur_all_sa_or = x_cur_all.mul(cur_all_sa)
        aux_fusion2 = weight*aux_fusion
        aux_fusion = aux_fusion+aux_fusion2+cur_all_sa_or


        x_L1 = aux_fusion + x_cur
        # x_LocAndGlo = self.badecoder(x_LocAndGlo)

        return x_L1
class gate2(nn.Module):
    def __init__(self, cur_channel,dims=[96, 192, 384, 768]):
        super(gate2, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()

        self.downconv = nn.Conv2d(dims[0],dims[0],kernel_size=1,padding=0,stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    def forward(self, x_cur,aux_1, aux_2, aux_3):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur_1)
        x_cur_3 = self.cur_b3(x_cur_2)
        x_cur_4 = self.cur_b4(x_cur_3)
        x_cur_all = self.cur_all(x_cur_4)
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = self.cur_all_sa(cur_all_ca)

        # aux1 conv
        aux_1 = self.downconv(aux_1)
        weight1 = self.sa1(aux_1)
        aux_1 = x_cur_all*weight1

        #aux_2 conv
        aux_2 = self.upsample2(aux_2)
        weight2 = self.sa2(aux_2)
        aux_2 = x_cur_all * weight2

        #aux_3 conv
        aux_3 = self.upsample4(aux_3)
        weight3 = self.sa3(aux_3)
        aux_3 = x_cur_all * weight3

        aux_fusion = aux_1 + aux_2 + aux_3

        weight = 1 - cur_all_sa
        cur_all_sa_or = x_cur_all.mul(cur_all_sa)
        aux_fusion2 = weight * aux_fusion
        aux_fusion = aux_fusion+aux_fusion2+cur_all_sa_or

        x_L2 = aux_fusion + x_cur
        # x_LocAndGlo = self.badecoder(x_LocAndGlo)

        return x_L2
class gate3(nn.Module):
    def __init__(self, cur_channel,dims=[96, 192, 384, 768]):
        super(gate3, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()

        self.downconv1_1 = nn.Conv2d(dims[0],dims[0],kernel_size=1,padding=0,stride=2)
        self.downconv1_2 = nn.Conv2d(dims[0], dims[0], kernel_size=1, padding=0, stride=2)
        self.downconv2 = nn.Conv2d(dims[1], dims[1], kernel_size=1, padding=0, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x_cur,aux_1, aux_2, aux_3):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur_1)
        x_cur_3 = self.cur_b3(x_cur_2)
        x_cur_4 = self.cur_b4(x_cur_3)
        x_cur_all = self.cur_all(x_cur_4)
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = self.cur_all_sa(cur_all_ca)

        # aux1 conv
        aux_1 = self.downconv1_1(aux_1)
        aux_1 = self.downconv1_2(aux_1)
        weight1 = self.sa1(aux_1)
        aux_1 = x_cur_all*weight1

        #aux_2 conv
        aux_2 = self.downconv2(aux_2)
        weight2 = self.sa2(aux_2)
        aux_2 = x_cur_all * weight2

        #aux_3 conv
        aux_3 = self.upsample2(aux_3)
        weight3 = self.sa3(aux_3)
        aux_3 = x_cur_all * weight3

        aux_fusion = aux_1 + aux_2 + aux_3

        weight = 1 - cur_all_sa
        cur_all_sa_or = x_cur_all.mul(cur_all_sa)
        aux_fusion2 = weight * aux_fusion
        aux_fusion = aux_fusion+aux_fusion2+cur_all_sa_or

        x_L3 = aux_fusion + x_cur
        # x_LocAndGlo = self.badecoder(x_LocAndGlo)

        return x_L3
class gate4(nn.Module):
    def __init__(self, cur_channel,dims=[96, 192, 384, 768]):
        super(gate4, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()

        self.downconv1_1 = nn.Conv2d(dims[0],dims[0],kernel_size=1,padding=0,stride=2)
        self.downconv1_2 = nn.Conv2d(dims[0], dims[0], kernel_size=1, padding=0, stride=2)
        self.downconv1_3 = nn.Conv2d(dims[0], dims[0], kernel_size=1, padding=0, stride=2)
        self.downconv2_1 = nn.Conv2d(dims[1], dims[1], kernel_size=1, padding=0, stride=2)
        self.downconv2_2 = nn.Conv2d(dims[1], dims[1], kernel_size=1, padding=0, stride=2)
        self.downconv3_1 = nn.Conv2d(dims[2], dims[2], kernel_size=1, padding=0, stride=2)



    def forward(self, x_cur,aux_1, aux_2, aux_3):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur_1)
        x_cur_3 = self.cur_b3(x_cur_2)
        x_cur_4 = self.cur_b4(x_cur_3)
        x_cur_all = self.cur_all(x_cur_4)
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = self.cur_all_sa(cur_all_ca)

        # aux1 conv
        aux_1 = self.downconv1_1(aux_1)
        aux_1 = self.downconv1_2(aux_1)
        aux_1 = self.downconv1_3(aux_1)
        weight1 = self.sa1(aux_1)
        aux_1 = x_cur_all*weight1

        #aux_2 conv
        aux_2 = self.downconv2_1(aux_2)
        aux_2 = self.downconv2_2(aux_2)
        weight2 = self.sa2(aux_2)
        aux_2 = x_cur_all * weight2

        #aux_3 conv
        aux_3 = self.downconv3_1(aux_3)
        weight3 = self.sa3(aux_3)
        aux_3 = x_cur_all * weight3

        aux_fusion = aux_1 + aux_2 + aux_3

        weight = 1 - cur_all_sa
        cur_all_sa_or = x_cur_all.mul(cur_all_sa)
        aux_fusion2 = weight * aux_fusion
        aux_fusion = aux_fusion+aux_fusion2+cur_all_sa_or

        x_L4 = aux_fusion + x_cur
        # x_LocAndGlo = self.badecoder(x_LocAndGlo)
        return x_L4

class SP(nn.Module):
    def __init__(self, channel1,channel2,channel3, dilation_1=2, dilation_2=4,dilation_3 = 8):
        super(SP, self).__init__()

        self.conv1 = BasicConv2d(channel1, channel2, 1, padding=0)
        self.conv1_Dila = BasicConv2d(channel2, channel2, 3, padding=dilation_1, dilation=dilation_1)

        self.conv2 = BasicConv2d(channel1, channel2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel2, channel2, 3, padding=dilation_2, dilation=dilation_2)

        self.conv3 = BasicConv2d(channel1, channel2, 5, padding=2)
        self.conv3_Dila = BasicConv2d(channel2, channel2, 3, padding=dilation_3, dilation=dilation_3)

        self.conv_fuse = BasicConv2d(channel2*3, channel3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x)
        x3_dila = self.conv3_Dila(x3)

        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3_dila), 1))
        # x_fuse = self.gcm1(x2)
        return x_fuse

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class decoder(nn.Module):
    def __init__(self, dims=[96, 192, 384, 768],nclass=None):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder4 = nn.Sequential(
            SP(dims[3],dims[3],dims[2]),
            # nn.Dropout(0.5),
            TransBasicConv2d(dims[2], dims[2], kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(dims[2], nclass, 3, stride=1, padding=1)


        self.decoder3 = nn.Sequential(
            SP(2*dims[2], dims[2], dims[1]),
            # nn.Dropout(0.5),
            TransBasicConv2d(dims[1], dims[1], kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(dims[1], nclass, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            SP(2*dims[1], dims[1], dims[0]),
            # nn.Dropout(0.5),
            TransBasicConv2d(dims[0], dims[0], kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(dims[0], nclass, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            SP(2*dims[0], dims[0], 64),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.S1 = nn.Conv2d(64, nclass, 3, stride=1, padding=1)


    def forward(self,x4, x3, x2, x1):
        x4_up = self.decoder4(x4)
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        x1_up = self.upsample(x1_up)
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4


class fusion(nn.Module):
    def __init__(self,inchannel):
        super(fusion, self).__init__()

        self.conv_w = BasicConv2d(inchannel,inchannel,kernel_size=3,padding=1)
    def forward(self,R,D):
        weight = R-D
        weight = self.conv_w(weight)
        r1 = weight*R
        d1 = weight*D
        r2 = r1+R
        d2 = d1+D
        out = r2+d2
        return out

class MGCNet(nn.Module):
    def __init__(self,dims=[96, 192, 384, 768],nclass=41):
        super(MGCNet, self).__init__()
        #Backbone model
        self.convnext_rgb = convnext_small(in_chans=3,pretrained=True)
        self.convnext_depth = convnext_small(in_chans=3,pretrained=True)
        self.norm1 = LayerNorm(dims[0], data_format='channels_first')
        self.norm2 = LayerNorm(dims[1], data_format='channels_first')
        self.norm3 = LayerNorm(dims[2], data_format='channels_first')
        self.norm4 = LayerNorm(dims[3], data_format='channels_first')
        self.fusions = nn.Sequential(
            fusion(inchannel=dims[0]),
            fusion(inchannel=dims[1]),
            fusion(inchannel=dims[2]),
            fusion(inchannel=dims[3]),
        )
        # Lateral layers
        # self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        # self.lateral_conv1 = BasicConv2d(256, 128, 3, stride=1, padding=1)
        # self.lateral_conv2 = BasicConv2d(512, 256, 3, stride=1, padding=1)
        # self.lateral_conv3 = BasicConv2d(1024, 512, 3, stride=1, padding=1)

        self.ACCoM4 = gate4(dims[3])
        self.ACCoM3 = gate3(dims[2])
        self.ACCoM2 = gate2(dims[1])
        self.ACCoM1 = gate1(dims[0])

        # self.agg2_rgbd = aggregation(channel)
        self.nclass = nclass
        self.decoder_rgb = decoder(nclass=self.nclass)

        # self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        # self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_end = nn.Conv2d(64,self.nclass,kernel_size=1,padding=0)



    def forward(self, rgb,depth):
        rgb_s1, rgb_s2, rgb_s3, rgb_s4 = self.convnext_rgb(rgb)
        rgb_s1 = self.norm1(rgb_s1)
        rgb_s2 = self.norm2(rgb_s2)
        rgb_s3 = self.norm3(rgb_s3)
        rgb_s4 = self.norm4(rgb_s4)
        # depth = depth[:,:1,:,:]
        depth_s1, depth_s2, depth_s3, depth_s4 = self.convnext_depth(depth)
        depth_s1 = self.norm1(depth_s1)
        depth_s2 = self.norm2(depth_s2)
        depth_s3 = self.norm3(depth_s3)
        depth_s4 = self.norm4(depth_s4)

        x4 = self.fusions[3](rgb_s4,depth_s4)
        x3 = self.fusions[2](rgb_s3,depth_s3)
        x2 = self.fusions[1](rgb_s2,depth_s2)
        x1 = self.fusions[0](rgb_s1,depth_s1)



        # up means update
        x4_ACCoM = self.ACCoM4(x4,x1,x2,x3)
        x3_ACCoM = self.ACCoM3(x3,x1, x2, x4)
        x2_ACCoM = self.ACCoM2(x2,x1, x3, x4)
        x1_ACCoM = self.ACCoM1(x1, x2,x3,x4)

        s1, s2, s3,s4 = self.decoder_rgb(x4_ACCoM, x3_ACCoM, x2_ACCoM, x1_ACCoM)
        # At test phase, we can use the HA to post-processing our saliency map

        if self.training:
            return s1, s2, s3,s4
        else:
            return s1
if __name__ == '__main__':
    x = torch.randn(1, 3, 480, 640)
    y = torch.randn(1, 3, 480, 640)
    net = MGCNet(nclass=41)
    # print(list(net.parameters())[0])
    # print(net.named_parameters())
    net1 = net(x, y)


    from toolbox.models.MGCNet.FLOP import CalParams

    CalParams(net, x, y)
    print(sum(p.numel() for p in net.parameters()) / 1000000.0)