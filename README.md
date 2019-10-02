# Kaggle: SIIM-ACR Pneumothorax Segmentation ([link](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation))

__Data__: 10 675 chest X-ray images

__Task__: predict the existence of pneumothorax in an image and indicate the location of the condition

__Evaluation__: mean Dice Coefficient

__Solution__: EfficientNet-ResNet U-Net++ (encoder-decoder network with nested, dense skip pathways)

__Success__: 0.841 mean Dice Coefficient

![](predictions.png)
