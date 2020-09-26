import iseg.types as T

from ._intersect_and_union import _intersect_and_union


def iou(
    segmap_list: T.List[T.Array], mask_list: T.List[T.Array], num_classes: int, ignore_index: int
) -> T.Dict[str, float]:

    """
    Compute the IoU over all predictions and masks in validataion or test dataset.

    Args:
        segmap_list (T.List[T.Array]): List of prediction segmentation maps.
        mask (T.List[T.Array]): List of gruond truth segmentation maps.
        num_classes (int): Number of classes.
        ignore_index (int): Index that should be ignored in evaluation.

    Returns:
        iou_dict (T.Dict[str, float]):
            iou_dict["mean"]: Average IoU for all classes.
            iou_dict[class_id]: IoU per class.

    References:
        https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/evaluation/mean_iou.py
    """

    total_area_dict = _intersect_and_union(segmap_list, mask_list, num_classes, ignore_index)
    iou_per_class = total_area_dict["intersect"] / total_area_dict["union"]

    iou_dict = {}
    iou_dict["mean"] = iou_per_class.mean()
    for class_id, class_iou in enumerate(iou_per_class):
        iou_dict[str(class_id)] = class_iou

    return iou_dict
