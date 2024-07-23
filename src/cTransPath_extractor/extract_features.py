from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset
from tqdm import tqdm

from pathlib import Path
import os

from timm.models.layers.helpers import to_2tuple
import torch.nn as nn


OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import torch
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import torch
from torchvision.models.resnet import Bottleneck, ResNet
import timm
from torchvision import transforms

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model

class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class TilesDataset(Dataset):
    def __init__(self, slide: openslide.OpenSlide, tiles_coords: np.ndarray) -> None:
        self.slide = slide
        self.tiles_coords = tiles_coords
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )  
        self.dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)
        file_extension = Path(self.slide._filename).suffix
        if file_extension == ".svs":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".qptiff":
            r = (
                ET.fromstring(slide.properties["openslide.comment"])
                .find("ScanProfile")
                .find("root")
                .find("ScanResolution")
            )
            self.magnification = float(r.find("Magnification").text)
        elif file_extension == ".ndpi":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".tiff": # experimental support of ome.tiff
            root = ET.fromstring(slide.properties["openslide.comment"])
            self.magnification = int(float(root[0][0].attrib["NominalMagnification"]))
        else:
            try :
                self.magnification = int(self.slide.properties["openslide.objective-power"])
            except:
                raise ValueError(f"File extension {file_extension} not supported")
        # We want the second highest level so as to have 112 microns tiles / 0.5 microns per pixel
        if self.magnification == 20:
            self.level = self.dz.level_count - 1
        elif self.magnification == 40:
            self.level = self.dz.level_count - 2
            self.magnification = 20
        else:
            raise ValueError(f"Objective power {self.magnification}x not supported")
        self.z = self.level

        assert np.all(
            self.tiles_coords[:, 0] == self.z
        ), "The resolution of the tiles is not the same as the resolution of the slide."

    def __getitem__(self, item: int):
        tile_coords = self.tiles_coords[item, 1:3].astype(int)
        try:
            im = self.dz.get_tile(level=self.level, address=tile_coords)
        except ValueError:
            print(f"ValueError: impossible to open tile {tile_coords} from {self.slide}")
            raise ValueError


        # im = ToTensor()(im)
        # if im.shape != torch.Size([3, 224, 224]):
        #     print(f"Image shape is {im.shape} for tile {tile_coords}. Padding...")
        #     # PAD the image in white to reach 224x224
        #     im = torch.nn.functional.pad(im, (0, 224 - im.shape[2], 0, 224 - im.shape[1]), value=1)

        im = self.transform(im)
        return im

    def __len__(self) -> int:
        return len(self.tiles_coords)



def get_features(
    slide: openslide.OpenSlide,
    model: torch.nn.Module,
    tiles_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    prefetch_factor: int = 8,
) -> np.ndarray:
    dataset = TilesDataset(slide=slide, tiles_coords=tiles_coords)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # num_workers=0 is necessary when using windows
        pin_memory=True ,
        drop_last=False
    )

    features = []
    dtype = next(model.parameters()).dtype
    for batch in tqdm(dataloader, leave = False):
        features_b= model(batch.type(dtype).to(device))
        features_b = features_b.half().cpu().detach()
        features.append(features_b)
    
    features = torch.concat(features).cpu().numpy()
    
    return features


def extract_features(
    slide_path: Path,
    device: torch.device,
    batch_size: int = 32,
    outdir: Path = None,
    tiles_coords_path: Path = None,
    tiles_coords: np.ndarray = None,
    num_workers: int = 0,
    pred_threshold: float = None,
    pred_comp: float=None,
    checkpoint_path: Path = None
):
    assert (
        tiles_coords_path is not None or tiles_coords is not None
    ), "Either tiles_coords_path or tiles_coords must be provided"

    slide = openslide.OpenSlide(str(slide_path))
    if tiles_coords is None:
        tiles_coords = np.load(tiles_coords_path)

#    model = timm.create_model(
#            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
#    )
#    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
#    model.eval()

    model = ctranspath()
    model.head = nn.Identity()
    #td = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['model'], strict=True)
    model.eval()

    
    #if torch.cuda.is_available():
    model = model.to(device)

    features = get_features(
        slide=slide,
        model=model,
        tiles_coords=tiles_coords,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return features
