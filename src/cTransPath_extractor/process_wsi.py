import os
import numpy as np
import h5py

# Demerdez-vous pour installer openslide sur votre machine
OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from display_results import display_wsi_results
from extract_tiles import filter_tiles
from extract_features import extract_features


from pathlib import Path
from torch import device

from argparse import ArgumentParser

from torch import device


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=Path(r"D:\PACPaint_homemade\temp_folder"),
        help="Path to the temporary directory where the features will be saved",
        required=True,
    )
    parser.add_argument(
        "--wsi",
        type=Path,
        default=Path(r"D:\PACPaint_homemade\datasets\HES_PAC_MULTICENTRIC_Ambroise Pare_ok\B00127107-002.svs"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(r"..\models\model_neo.pth"),
        help="Path to the cTransPath model",
    )
    parser.add_argument(
        "--pred_tiles",
        type=Path,
        default=Path(r"..\models\model_comp_BASAL_CLASSIC_only.pth"),
        help="Path to the csv of tiles prediction",
    )
    parser.add_argument(
        "--device", type=device, default="cuda:0", help="Device to use for the predictions"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for the feature extraction")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the feature extraction. Set to 0 if using windows.",
    )
    parser.add_argument(
        "--comp_threshold",
        type=float,
        default=0.5,
        help="Threshold fro the prediction tumor cells",
    )
    parser.add_argument("--pred_threshold", type=float, default=0.35, help="Threshold for the predictions tumor")
    
    parsed_args = parser.parse_args()
    return parser.parse_args()


def main(args):
    slidename = args.wsi.stem
    print("Filtering tiles...")
    tiles_coord = filter_tiles(args.wsi, args.pred_tiles, args.pred_threshold,args.comp_threshold)

    print("Extracting stroma features...")
    features_stroma = extract_features(
        args.wsi,
        args.device,
        args.batch_size,
        outdir = args.temp_dir,
        tiles_coords=tiles_coord["stroma"],
        num_workers=args.num_workers,
        checkpoint_path = args.model_path
    )
    with h5py.File(f'{args.temp_dir}/{slidename}_cTransPath_stroma.h5', 'w') as f:
        f['coords'] = tiles_coord["stroma"]
        f['feats'] = features_stroma
        f['type'] = 'stroma'
        f['extractor'] = 'cTransPath'


    print("Extracting tumoral features...")
    features_tum = extract_features(
        args.wsi,
        args.device,
        args.batch_size,
        outdir = args.temp_dir,
        tiles_coords=tiles_coord["tumCells"],
        num_workers=args.num_workers,
        checkpoint_path = args.model_path
    )
    with h5py.File(f'{args.temp_dir}/{slidename}_cTransPath_tumor.h5', 'w') as f:
        f['coords'] = tiles_coord["tumCells"]
        f['feats'] = features_tum
        f['type'] = 'tumor_cell'
        f['extractor'] = 'cTransPath'

    print("Done")



if __name__ == "__main__":
    args = parse_arg()
    main(args)
