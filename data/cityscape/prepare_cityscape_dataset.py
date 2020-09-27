import os
from pathlib import Path

import cv2
import iseg.types as T
import numpy as np
import pandas as pd
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from tqdm import tqdm


def create_info_csv(base: T.Path) -> T.DataFrame:

    os.mkdir(base / "images")
    os.mkdir(base / "masks")

    di: dict = {
        "old_img_stem": [],
        "json_file": [],
        "city_name": [],
        "data_type": [],
        "stem": [],
    }
    for p in base.glob("leftImg8bit/*/*/*_leftImg8bit.png"):
        city_name = p.parents[0].stem
        data_type = p.parents[1].stem
        stem = "_".join(p.stem.split("_")[:3])
        stem = f"{data_type}_{stem}"
        old_img_stem = p.stem
        json_file = "_".join(p.stem.split("_")[:3])
        json_file = f"{json_file}_gtFine_polygons.json"

        di["city_name"].append(city_name)
        di["data_type"].append(data_type)
        di["stem"].append(stem)
        di["old_img_stem"].append(old_img_stem)
        di["json_file"].append(json_file)

    df = pd.DataFrame(di)
    df.to_csv(base / "info.csv", index=False)
    return df


def move_images(base: T.Path, df: T.DataFrame) -> None:

    pbar = tqdm(df.index, desc="move_images")
    for i in pbar:
        old_img_stem = df.loc[i, "old_img_stem"]
        stem = df.loc[i, "stem"]
        data_type = df.loc[i, "data_type"]
        city_name = df.loc[i, "city_name"]

        img_src = base / f"leftImg8bit/{data_type}/{city_name}/{old_img_stem}.png"
        img_dst = base / f"images/{stem}.png"

        os.rename(img_src, img_dst)


def convert_json_to_mask(base: T.Path, df: T.DataFrame) -> None:

    pbar = tqdm(df.index, desc="convert_json_to_mask")
    for i in pbar:
        data_type = df.loc[i, "data_type"]
        city_name = df.loc[i, "city_name"]
        json_file = df.loc[i, "json_file"]
        stem = df.loc[i, "stem"]

        json_path = base / f"gtFine/{data_type}/{city_name}/{json_file}"
        mask_path = base / f"masks/{stem}.png"
        json2labelImg(json_path, mask_path, "trainIds")


def convert_png_to_npy(base: T.Path) -> None:

    pbar = tqdm(base.glob("images/*.png"), desc="convert_png_to_npy (images)")
    for p in pbar:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with open(p.parent / f"{p.stem}.npy", "wb") as f:
            np.save(f, img)
        p.unlink()

    pbar = tqdm(base.glob("masks/*.png"), desc="convert_png_to_npy (masks)")
    for p in pbar:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        with open(p.parent / f"{p.stem}.npy", "wb") as f:
            np.save(f, mask)
        p.unlink()


if __name__ == "__main__":

    base = Path("data/cityscape")
    df = create_info_csv(base)
    move_images(base, df)
    convert_json_to_mask(base, df)
    convert_png_to_npy(base)
