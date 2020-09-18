import os
from pathlib import Path

import pandas as pd


def create_info_csv(base: Path) -> pd.DataFrame:

    os.mkdir(base / "images")
    os.mkdir(base / "masks")

    di: dict = {
        "old_img_stem": [],
        "old_mask_stem": [],
        "label_name": [],
        "data_type": [],
        "stem": [],
    }
    for p in base.glob("leftImg8bit/*/*/*_leftImg8bit.png"):
        label_name = p.parents[0].stem
        data_type = p.parents[1].stem
        stem = "_".join(p.stem.split("_")[:3])
        stem = f"{data_type}_{stem}"
        old_img_stem = p.stem
        old_mask_stem = "_".join(p.stem.split("_")[:3])
        old_mask_stem = f"{old_mask_stem}_gtFine_labelIds"

        di["label_name"].append(label_name)
        di["data_type"].append(data_type)
        di["stem"].append(stem)
        di["old_img_stem"].append(old_img_stem)
        di["old_mask_stem"].append(old_mask_stem)

    return pd.DataFrame(di)


def move_images_and_masks(base: Path, df: pd.DataFrame):

    for i in df.index:
        old_img_stem = df.loc[i, "old_img_stem"]
        old_mask_stem = df.loc[i, "old_mask_stem"]
        stem = df.loc[i, "stem"]
        data_type = df.loc[i, "data_type"]
        label_name = df.loc[i, "label_name"]

        img_src = base / f"leftImg8bit/{data_type}/{label_name}/{old_img_stem}.png"
        img_dst = base / f"images/{stem}.png"
        mask_src = base / f"gtFine/{data_type}/{label_name}/{old_mask_stem}.png"
        mask_dst = base / f"masks/{stem}.png"

        os.rename(img_src, img_dst)
        os.rename(mask_src, mask_dst)


if __name__ == "__main__":

    base = Path("data/cityscape")
    df = create_info_csv(base)
    move_images_and_masks(base, df)
