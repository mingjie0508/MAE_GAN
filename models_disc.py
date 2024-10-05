# GAN discriminator

import timm
import torch.nn as nn


class Discriminator(nn.Module):
    REAL = 1.0
    FAKE = 0.0

    def __init__(self, model_name, num_classes=1):
        super(Discriminator, self).__init__()
        self.discriminator = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.discriminator(x)
    
    def get_loss(self, output, label):
        return self.criterion(output, label)


if __name__ == "__main__":
    model = Discriminator('vit_tiny_patch16_224')
    # model = Discriminator('vit_small_patch16_224')
    # model = Discriminator('vit_base_patch16_224')
    print(model)
