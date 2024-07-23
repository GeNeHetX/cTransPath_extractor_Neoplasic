import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import os

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def convert_coord(img, slide, df_slide):
    h, w = img.shape[:2]
    w_slide, h_slide = slide.dimensions
    h_factor, w_factor = h / h_slide, w / w_slide
    df_slide["x_img"] = df_slide.x * w_factor * 2 * 224
    df_slide["y_img"] = df_slide.y * h_factor * 2 * 224
    return df_slide


def display_wsi_results(
    path_wsi: Path,
    pred_csv: Path,
    neo_threshold: float = 0.35,
    tumor_threshold: float=0.5,
    display: bool=False,
    outdir: Path=None
) -> None:
    slide = openslide.OpenSlide(str(path_wsi))
    img_pil = slide.get_thumbnail((1000, 1000))
    img = np.array(img_pil)

    pred_df = pd.read_csv(pred_csv)

    pred_df = convert_coord(img, slide, pred_df) 

    pred_df['Color'] = pred_df.apply(
        lambda row: 1 if row['pred_tumor'] < neo_threshold
               else 2 if row['pred_tumor'] >= neo_threshold 
               and row['pred_tumor_cell'] < tumor_threshold else 3, axis=1)
    
    color_mappings = {1: 'grey', 2: 'red', 3: 'blue'}
    color_labels = {1: 'No Neoplasic', 2: 'Stroma', 3: 'Tumor'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten()
    ax1, ax2 = axes
    ax1.imshow(img)
    ax1.set_title("Original image")
    for color_label in set(pred_df.Color):
        color_data = pred_df[pred_df.Color == color_label]
        ax2.scatter(color_data.x_img, color_data.y_img, s=2, marker='s', c=color_mappings[color_label], label=color_labels[color_label])

    ax2.invert_yaxis()
    ax2.set_title("Tumor prediction")
    

    # Both subplot same size
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
    plt.tight_layout()
    plt.legend(loc="upper right")
    if display:
        plt.show()

    plt.savefig(f"{outdir}/{path_wsi.stem}.png")


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--path_wsi",
        type=Path,
        help="Path to the Whole Slide Image",
        required=True,
    )
    parser.add_argument(
        "--pred_csv",
        type=Path,
        required=True,
        help="Path to the csv of tiles prediction",
    )
    parser.add_argument(
        "--neo_threshold",
        type=float,
        default=0.35,
        help="Threshold for the neoplasic prediction",
    )
    parser.add_argument("--tumor_threshold",
        type=float, 
        default=0.5, 
        help="Threshold for the tumor cell prediction"
    )
    parser.add_argument("--display", action="store_true", help="Display the WSI and the tiles")
    parser.add_argument("--out_dir",
        type=Path, 
        help="Path to output directory"
    )
    
    parsed_args = parser.parse_args()
    return parser.parse_args()



def main(args):

    slidename = args.path_wsi.stem
    print(f"Process {slidename}...")

    display_wsi_results(
        path_wsi=args.path_wsi,
        pred_csv=args.pred_csv,
        neo_threshold=args.neo_threshold,
        tumor_threshold=args.tumor_threshold,
        display=args.display,
        outdir=args.out_dir
    )
    



if __name__ == "__main__":

    args = parse_arg()
    main(args)
