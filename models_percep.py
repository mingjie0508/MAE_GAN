# Perceptual loss model
# computes loss over deep features of a pre-trained CNN

import torch.nn as nn
import timm

class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        model_name = 'vgg16_bn'
        blocks = []
        blocks.append(timm.create_model(model_name, pretrained=True).features[:6].eval())
        blocks.append(timm.create_model(model_name, pretrained=True).features[6:13].eval())
        blocks.append(timm.create_model(model_name, pretrained=True).features[13:23].eval())
        blocks.append(timm.create_model(model_name, pretrained=True).features[23:33].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.criterion = nn.L1Loss()
    
    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        # input = x (original image)
        # target = fake_x (generated image)
        # Both have shape [batch_size, 3, 224, 224]
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)  
            if i in feature_layers:  # i = 0, 1, 2, 3
                loss += self.criterion(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, H*W)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)  # (B, C, C)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += self.criterion(gram_x, gram_y)
        return loss
