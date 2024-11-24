# MAE_GAN

In this project, we explore the use of Masked Autoencoder (MAE) in inpainting tasks. By combining MAE with GAN, we can obtain clear, detailed inpainted images.

## Installation

```
pip install -r requirements.txt
```

## Dataset
[PASCAL VOC 2012](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)

17095 RGB images, 80% for training, 20% for testing

## Run
```train_gan.ipynb```: training MSE-GAN

```train_ganpercep.ipynb```: training MAE-GAN + perceptual loss

```train_ganbound.ipynb```: training MAE-GAN + boundary loss

```train_ganbound_percep.ipynb```: training MAE-GAN + boundary loss + perceptual loss

```train_iprompt.ipynb```: training MAE-GAN + boundary loss + perceptual loss for image prompting