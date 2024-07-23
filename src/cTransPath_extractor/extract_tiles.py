import numpy as np
from tqdm import tqdm
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset
from skimage.morphology import disk, binary_closing
from scipy.ndimage import binary_fill_holes
from pathlib import Path
import os
import pandas as pd 

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import xml.etree.ElementTree as ET


class TilesWhiteDataset(Dataset):
    def __init__(
        self,
        slide: openslide.OpenSlide,
        tile_size: int = 224,
    ) -> None:
        self.slide = slide
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
            self.magnification = int(r.find("Magnification").text)
        elif file_extension == ".ndpi":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        else:
            raise ValueError(f"File extension {file_extension} not supported")
        self.dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
        # We want the second highest level so as to have 112 microns tiles / 0.5 microns per pixel
        if self.magnification == 20:
            self.level = self.dz.level_count - 1
        elif self.magnification == 40:
            self.level = self.dz.level_count - 2
        else:
            raise ValueError(f"Objective power {self.magnification}x not supported")
        self.h, self.w = self.dz.level_dimensions[self.level]
        self.h_tile, self.w_tile = self.dz.level_tiles[self.level]
        # Get rid of the last row and column because they can't fit a full tile usually
        self.h_tile -= 1
        self.w_tile -= 1
        self.z = self.level

    def idx_to_ij(self, item: int):
        return np.unravel_index(item, (self.h_tile, self.w_tile))

    def __len__(self) -> int:
        return self.h_tile * self.w_tile


def filter_tiles(path_svs, pred_path, pred_threshold, pred_comp):
    tile_size = 224
    predictions = pd.read_csv(pred_path)

    tumor =  predictions[predictions["pred_tumor"]> pred_threshold]
    stroma = tumor[tumor["pred_tumor_cell"] < pred_comp]
    print(f"Finding {stroma.shape[0]} stroma tiles")
    stroma_tiles = stroma[["z","x","y","pred_tumor","pred_tumor_cell"]].to_numpy()
    tumorCell = tumor[tumor["pred_tumor_cell"] > pred_comp]
    print(f"Finding {tumorCell.shape[0]} tumoral cell tiles")
    tumor_tiles = tumorCell[["z","x","y","pred_tumor","pred_tumor_cell"]].to_numpy()


    return {'stroma': stroma_tiles, 'tumCells': tumor_tiles}
