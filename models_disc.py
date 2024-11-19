# GAN discriminator

import timm
import torch.nn as nn


class Discriminator(nn.Module):
    # discriminate over entire image
    REAL = 1.0
    FAKE = 0.0

    def __init__(self, model_name):
        super(Discriminator, self).__init__()
        self.discriminator = timm.create_model(
            model_name, pretrained=True, num_classes=1
        )
        # neighbouring patches overlap, helps with training
        self.overlap = 2
        self.discriminator.patch_embed.strict_img_size = False
        self.discriminator.patch_embed.proj = nn.Conv2d(
            3,
            self.discriminator.embed_dim,
            kernel_size=16+2*self.overlap,
            stride=16,
        )
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        # pad borders of input image, neighbouring patches overlap
        x = nn.functional.pad(
            x, (self.overlap, self.overlap, self.overlap, self.overlap), 
            mode='replicate'
        )
        return self.discriminator(x)
    
    def get_loss(self, output, label):
        return self.criterion(output, label)


class Discriminator2(nn.Module):
    # discriminate patch by patch
    REAL = 1.0
    FAKE = 0.0

    def __init__(self, model_name):
        super(Discriminator2, self).__init__()
        self.discriminator = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )
        self.head = nn.Linear(self.discriminator.embed_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        # pass input through ViT, and remove class token
        x = self.discriminator.forward_features(x)
        x = self.head(x)
        return x[:, 1:, 0]

    def get_loss(self, output, label):
        return self.criterion(output, label)


if __name__ == "__main__":
    model = Discriminator('vit_tiny_patch16_224')
    # model = Discriminator('vit_small_patch16_224')
    # model = Discriminator('vit_base_patch16_224')
    print(model)
