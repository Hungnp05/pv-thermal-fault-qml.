import timm
import torch.nn as nn

CNN_NAMES = {
    "mobilenetv3_small": "mobilenetv3_small_100",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
    "coat_lite_small": "coat_lite_small",
}
DROP_RATE = 0.2


# Create an ImageNet-pretrained backbone with a new classification head.
def build_cnn(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = DROP_RATE) -> nn.Module:
    return timm.create_model(CNN_NAMES[name], pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


# Freeze or unfreeze every layer except the classification head.
def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head = set(model.get_classifier().parameters())
    for parameter in model.parameters():
        if parameter not in head:
            parameter.requires_grad = trainable


# Count trainable and total parameters of a model.
def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
