# GAN discriminator

import timm
import torch.nn as nn


class GANDiscriminator(nn.Module):
    def __init__(self, model_name):
        super(GANDiscriminator, self).__init__()
        self.discriminator = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )
        # classification head for whole image
        self.head_i = nn.Linear(self.discriminator.embed_dim, 1)
        # classification head for patch by patch
        self.head = nn.Linear(self.discriminator.embed_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        # pass input through model
        x = self.discriminator.forward_features(x)
        # Shape: (batch_size, num_patches + 1, embed_dim)
        #   For 224x224 image with patch_size=16:
        #   - num_patches = (224/16)² = 196
        #   - +1 for cls_token
        #   So shape is (batch_size, 197, embed_dim)

        # discriminate whole image
        y_i = self.head_i(x)  # Shape: (batch_size, 197, 1)
        y_i = y_i[:, 0, 0]
        # discriminate patch by patch
        y_p = self.head(x)
        y_p = y_p[:, 1:, 0]   # Shape: (batch_size, 196)
        return y_i, y_p


if __name__ == "__main__":
    model = GANDiscriminator('vit_tiny_patch16_224')
    # model = GANDiscriminator('vit_small_patch16_224')
    # model = GANDiscriminator('vit_base_patch16_224')
    print(model)
