import math

import numpy as np
import torch
import torch.nn.functional as F


# Turn transformer tokens (batch, tokens, channels) into a square feature grid; conv feature maps pass through.
def to_grid(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    tokens = x if math.isqrt(x.shape[1]) ** 2 == x.shape[1] else x[:, 1:]
    side = math.isqrt(tokens.shape[1])
    return tokens.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)


# Pick the layer to explain: the second-to-last CoaT block, because its final patch tokens never reach the class token.
def target_layer(model: torch.nn.Module):
    blocks = getattr(model, "serial_blocks4", None)
    return blocks[-2] if blocks is not None else None


# Run the model once and return the logits together with the feature tensor used for Grad-CAM.
def forward_with_features(model: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    layer = target_layer(model)
    if layer is None:
        features = model.forward_features(batch)
        features.retain_grad()
        return model.forward_head(features), features
    store = {}
    keep = lambda module, inputs, output: store.update(features=output) or output.retain_grad()
    handle = layer.register_forward_hook(keep)
    logits = model(batch)
    handle.remove()
    return logits, store["features"]


# Return a HiResCAM heatmap in [0, 1] at input resolution showing where the model looked for the target class.
def grad_cam(model: torch.nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    model.eval()
    with torch.enable_grad():
        logits, features = forward_with_features(model, image.unsqueeze(0))
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
    maps, grads = to_grid(features.detach()), to_grid(features.grad)
    cam = F.relu((grads * maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp_min(1e-8)).cpu().numpy()
