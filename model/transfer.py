import torch
import torchvision.models as models
import timm
import os

# Number of classes in the dataset
classes = 5

# Load the ResNet50 model pretrained on the ImageNet dataset
resnet_model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
resnet_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


# Load the EfficientNetV2-Small model pretrained on the ImageNet dataset
efficientnet_model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
efficientnet_model.conv_stem = torch.nn.Conv2d(1, efficientnet_model.conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)

# ResNet50 linear layer
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, classes)

num_ftrs = efficientnet_model.classifier.in_features
efficientnet_model.classifier = torch.nn.Linear(num_ftrs, classes)

# create folder for models
os.makedirs(r'transfer_models', exist_ok=True)

# Export the models for import into other Python files
torch.save(resnet_model, r'transfer_models\resnet_model.pth')
torch.save(efficientnet_model, r'transfer_models\efficientnet_model.pth')
