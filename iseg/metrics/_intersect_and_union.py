import iseg.types as T
import numpy as np


def _intersect_and_union(
    segmap_list: T.List[T.Array], mask_list: T.List[T.Array], num_classes: int, ignore_index: int
) -> T.Dict[str, T.Array]:

    """
    Compute the total intersect and union area over all predictions and masks in validation or test dataset.

    Args:
        segmap_list (T.List[T.Array]): List of prediction segmentation maps.
        mask_list (T.List[T.Array]): List of gruond truth segmentation maps.
        num_classes (int): Number of classes.
        ignore_index (int): Index that should be ignored in evaluation.

    Returns:
        total_area_dict (T.Dict[T.Array]):
            total_area_dict["segmap"]: Total area of prediction segmentation maps per class.
            total_area_dict["mask"]: Total area of ground truth segmentation maps per class.
            total_area_dict["intersect"]: Total area of intersect per class
            total_area_dict["union"]: Total area of union per class

    References:
        https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/evaluation/mean_iou.py
    """

    zero_array = np.zeros(num_classes, dtype=np.float)
    total_area_dict = {
        "segmap": zero_array.copy(),
        "mask": zero_array.copy(),
        "intersect": zero_array.copy(),
        "union": zero_array.copy(),
    }

    for segmap, mask in zip(segmap_list, mask_list):

        # segmap.shape -> (h, w), mask.shape -> (h, w)
        bool_array = mask != ignore_index
        segmap = segmap[bool_array]
        mask = mask[bool_array]
        intersect = segmap[segmap == mask]

        bins = np.arange(num_classes + 1)
        segmap_area, _ = np.histogram(segmap, bins=bins)
        mask_area, _ = np.histogram(mask, bins=bins)
        intersect_area, _ = np.histogram(intersect, bins=bins)
        union_area = segmap_area + mask_area - intersect_area

        total_area_dict["segmap"] += segmap_area
        total_area_dict["mask"] += mask_area
        total_area_dict["intersect"] += intersect_area
        total_area_dict["union"] += union_area

    return total_area_dict
