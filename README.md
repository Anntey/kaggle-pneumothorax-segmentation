# Kaggle: SIIM-ACR Pneumothorax Segmentation

Data: 10 675 chest X-ray images

Task: predict the existence of pneumothorax in an image and indicate the location of the condition

Evaluation: mean Dice coefficient

Solution: encoder-decoder (EfficientNet-ResNet) network with nested, dense skip pathways

Success: 0.841 mean Dice coefficient

![](preditions.png)
