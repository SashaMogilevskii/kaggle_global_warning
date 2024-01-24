import loguru
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, backbone_name, in_channels, weight=None):
        super().__init__()

        self.encoder = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=weight,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
        # print(self.encoder.encoder.patch_embed1.proj)
        # out_channels = self.encoder.encoder.patch_embed1.proj.out_channels
        # # self.encoder.encoder.patch_embed1.proj = nn.Conv2d(in_channels, out_channels, 7, 4, 3)

    def forward(self, images: torch.Tensor):
        output = self.encoder(images)
        return output


class CustomModelDeepLabV3Plus(nn.Module):
    def __init__(self, backbone_name, in_channels, weight=None):
        super().__init__()

        self.encoder = smp.DeepLabV3Plus(
            encoder_name=backbone_name,
            encoder_weights=weight,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
        # print(self.encoder.encoder.patch_embed1.proj)
        # out_channels = self.encoder.encoder.patch_embed1.proj.out_channels
        # # self.encoder.encoder.patch_embed1.proj = nn.Conv2d(in_channels, out_channels, 7, 4, 3)

    def forward(self, images: torch.Tensor):
        output = self.encoder(images)
        return output


class CustomModelUnetPlusPlus(nn.Module):
    def __init__(self, backbone_name, in_channels, encoder_depth,
                 decoder_channels, weight=None):
        super().__init__()

        self.encoder = smp.UnetPlusPlus(
            encoder_name=backbone_name,
            encoder_weights=weight,
            in_channels=in_channels,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            classes=1,
            activation=None,
        )
        # print(self.encoder.encoder.patch_embed1.proj)
        # out_channels = self.encoder.encoder.patch_embed1.proj.out_channels
        # # self.encoder.encoder.patch_embed1.proj = nn.Conv2d(in_channels, out_channels, 7, 4, 3)

    def forward(self, images: torch.Tensor):
        output = self.encoder(images)
        return output


import torch
from torch import nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.relu(x)
        x = self.conv2(x)

        x = self.relu(x)
        return x


class TransUnet(nn.Module):
    """
        Custom model TransUnet
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model('pvt_v2_b5', pretrained=True, num_classes=1, in_chans=in_channels,
                                         features_only=True)
        self.conv_block1 = ConvBlock(64, 64)
        self.conv_block2 = ConvBlock(128, 128)
        self.conv_block3 = ConvBlock(320, 320)
        self.conv_block4 = ConvBlock(512, 512)
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_block5 = ConvBlock(832, 320)
        self.conv_block6 = ConvBlock(448, 128)
        self.conv_block7 = ConvBlock(192, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_block8 = ConvBlock(64, 64)

    def forward(self, x):
        out = self.encoder(x)

        out = [self.conv_block1(out[0]), self.conv_block2(out[1]), self.conv_block3(out[2]), self.conv_block4(out[3])]

        x = out[3]

        x = self.up1(x)
        x = torch.cat((x, out[2]), dim=1)
        x = self.conv_block5(x)

        x = self.up2(x)
        x = torch.cat((x, out[1]), dim=1)
        x = self.conv_block6(x)

        x = self.up3(x)
        x = torch.cat((x, out[0]), dim=1)
        x = self.conv_block7(x)

        x = self.up4(x)
        x = self.up5(x)
        x = self.conv_block8(x)

        x = self.final_conv(x)

        return x


import timm


class ImprovedSemanticFPN(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedSemanticFPN, self).__init__()

        # Load the pretrained backbone encoder
        self.encoder = timm.create_model('pvt_v2_b5', pretrained=True, num_classes=1, in_chans=3,
                                         features_only=True)

        # Define your decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, num_classes, kernel_size=1)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        # Define any additional layers if needed

    def forward(self, x):
        # Pass input through the backbone encoder
        feats = self.encoder(x)

        # Retrieve feature maps from different levels of the backbone
        c1, c2, c3, c4 = feats[-4:]
        for i in feats[-4:]:
            print(i.shape)

        # Upsample and apply decoder layers
        p4 = self.decoder4(c4)
        p3 = self.decoder3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.decoder2(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self.decoder1(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)

        # Upsample the final prediction to the original input size
        output = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=False)

        return output


class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvSilu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class Timm_Unet(nn.Module):
    def __init__(self, name='coatnet_rmlp_2_rw_384', pretrained=True, inp_size=3, otp_size=1, decoder_filters=[32, 48, 64, 96, 128],
                 **kwargs):
        super(Timm_Unet, self).__init__()


        encoder = timm.create_model(name, pretrained=pretrained, in_chans=inp_size, features_only=True)


        encoder_filters = [f['num_chs'] for f in encoder.feature_info]

        decoder_filters = decoder_filters

        self.conv6 = ConvSilu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvSilu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvSilu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvSilu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvSilu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvSilu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvSilu(decoder_filters[-3], decoder_filters[-4])

        if len(encoder_filters) == 4:
            self.conv9_2 = None

        else:
            self.conv9_2 = ConvSilu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])

        self.conv10 = ConvSilu(decoder_filters[-4], decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], otp_size, 1, stride=1, padding=0)

        self.cls = nn.Linear(encoder_filters[-1] * 2, 5)
        self.pix_sz = nn.Linear(encoder_filters[-1] * 2, 1)

        self._initialize_weights()

        self.encoder = encoder

    def forward(self, x):
        batch_size, C, H, W = x.shape

        if self.conv9_2 is None:
            enc2, enc3, enc4, enc5 = self.encoder(x)

        else:
            enc1, enc2, enc3, enc4, enc5 = self.encoder(x)



        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                                       ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                                       ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                                       ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        if self.conv9_2 is not None:
            dec9 = self.conv9_2(torch.cat([dec9,
                                           enc1
                                           ], 1))

        dec10 = self.conv10(dec9)  # F.interpolate(dec9, scale_factor=2))



        # x1 = F.dropout(x1, p=0.3, training=self.training)


        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
if __name__ == '__main__':
    # encoder = generate_model(model_depth=50, n_input_channels=1)
    # pred = encoder.forward(n_batch)
    # model = Seg_Model_ResNet50(depth=101)
    # model = TransUnet(in_channels=16)
    # model = ImprovedSemanticFPN()

    # model = CustomModel(backbone_name="tu-coatnet_rmlp_2_rw_384", in_channels=3, weight='ImageNet')

    model = Timm_Unet()
    # encoder = timm.create_model('coatnet_rmlp_2_rw_384', pretrained=True, num_classes=1, in_chans=3,
    #                             features_only=True)

    n_batch = torch.rand((2, 3, 512, 512))

    pred = model.forward(n_batch)
    print(pred[0].shape)
    print(pred[1])
    print(pred[2])

    # torch.save(model, 'resnet101.pt')
