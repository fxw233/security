import torch.nn as nn
import torch
from t2t_vit import T2t_vit_t_14
from Transformer import Transformer
from Transformer import token_Transformer
from Decoder import Decoder
import numpy as np
import cv2
import transforms 
def dct(image_Input):
        image_Input = image_Input.cpu().numpy()
        res = []
        for img in image_Input:
            new_img = []
            for channel in img:
                f = np.fft.fft2(channel)
                fshift = np.fft.fftshift(f)
                rows, cols = channel.shape
                crow,ccol = rows//2 , cols//2
                fshift[crow-1:crow+1, ccol-1:ccol+1] = 0
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)

                new_img.append(np.array(img_back).astype(np.float32))
            res.append(np.array(new_img))
        return np.array(res)
        
class ImageDepthNet2(nn.Module):
    

    def __init__(self, args):
        super(ImageDepthNet2, self).__init__()


        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.dct_backbone = T2t_vit_t_14(pretrained=True, args=args)
        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        self.concatFuse = nn.Sequential(
                nn.Linear(384*2, 384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
        self.concatFuse2 = nn.Sequential(
                nn.Linear(64*2, 64),
                nn.GELU(),
                nn.Linear(64, 64),
            )

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape

        dct_input = torch.from_numpy(dct(image_Input)).cuda()
        
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)

        dct_1_16, dct_1_8, dct_1_4 = self.dct_backbone(dct_input)
        # VST Convertor
        rgb_fea_1_16 = self.concatFuse(torch.cat([rgb_fea_1_16, dct_1_16], dim=2))
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]
        rgb_fea_1_8 = self.concatFuse2(torch.cat([rgb_fea_1_8, dct_1_8], dim=2))
        rgb_fea_1_4 = self.concatFuse2(torch.cat([rgb_fea_1_4, dct_1_4], dim=2))
        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)

        return outputs
    