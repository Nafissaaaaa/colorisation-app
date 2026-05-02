import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
#  Architecture exacte de ton modèle Pix2Pix
# ─────────────────────────────────────────────

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, apply_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, apply_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """U-Net Pix2Pix: 1-channel grayscale in -> 3-channel RGB out (tanh, range [-1,1])."""
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.d1 = DownBlock(in_channels, 64,  apply_bn=False)
        self.d2 = DownBlock(64,  128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)
        self.d7 = DownBlock(512, 512)
        self.d8 = DownBlock(512, 512)

        self.u1 = UpBlock(512,        512, apply_dropout=True)
        self.u2 = UpBlock(512 + 512,  512, apply_dropout=True)
        self.u3 = UpBlock(512 + 512,  512, apply_dropout=True)
        self.u4 = UpBlock(512 + 512,  512)
        self.u5 = UpBlock(512 + 512,  256)
        self.u6 = UpBlock(256 + 256,  128)
        self.u7 = UpBlock(128 + 128,  64)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))
        return self.last(torch.cat([u7, d1], dim=1))


# ─────────────────────────────────────────────
#  Chargement du modèle
# ─────────────────────────────────────────────

def load_model(model_path: str, device: str = "cpu") -> Generator:
    """
    Charge le Generator depuis best_model(2).pt ou tout autre .pt/.pth.
    Gere 3 formats :
      1. state_dict direct  -> torch.save(generator.state_dict(), path)
      2. dict avec cle      -> torch.save({"state_dict": ...}, path)
      3. modele complet     -> torch.save(generator, path)
    """
    model = Generator()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        # Cas 1 : state_dict direct (cles comme "d1.block.0.weight", ...)
        if any(k.startswith("d1.") or k.startswith("u1.") for k in checkpoint.keys()):
            model.load_state_dict(checkpoint)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "generator" in checkpoint:
            model.load_state_dict(checkpoint["generator"])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Cas 3 : modele complet sauvegarde
        model = checkpoint

    model.eval()
    model.to(device)
    return model


# ─────────────────────────────────────────────
#  Pipeline de colorisation
# ─────────────────────────────────────────────

def preprocess(image: Image.Image, size: int = 256):
    """PIL image -> tenseur [-1,1] shape (1,1,H,W) + taille originale."""
    original_size = image.size
    gray = image.convert("L").resize((size, size))
    arr  = np.array(gray, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, original_size


def postprocess(pred: torch.Tensor, original_size: tuple) -> Image.Image:
    """Tenseur (1,3,H,W) tanh [-1,1] -> image PIL RGB."""
    img = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
    pil = Image.fromarray(img, mode="RGB")
    return pil.resize(original_size, Image.LANCZOS)


def colorize(image: Image.Image, model: Generator,
             device: str = "cpu", size: int = 256) -> Image.Image:
    """Pipeline complet : image N&B PIL -> image colorisée PIL."""
    tensor, original_size = preprocess(image, size=size)
    tensor = tensor.to(device)
    with torch.no_grad():
        pred = model(tensor)
    return postprocess(pred, original_size)
