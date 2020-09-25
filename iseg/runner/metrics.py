import iseg.metrics
import iseg.types as T


class RunnerMetrics:

    cfg: T.DictConfig
    log: T.Logger

    def compute_and_log_metrics(
        self, data_type: str, segmap_list: T.List[T.Tensor], mask_list: T.List[T.Tensor]
    ) -> None:

        """
        Compute the metrics specified by yaml file and then, log them.

        Args:
            data_type (str): Dataset type (val or test).
            segmap_list (T.List[T.Array]): List of prediction segmentation maps.
            mask (T.List[T.Array]): List of gruond truth segmentation maps.
        """

        for metric_cfg in self.cfg.metrics:

            compute_metric_fn = getattr(iseg.metrics, metric_cfg.name)
            metric_dict = compute_metric_fn(segmap_list, mask_list, **metric_cfg.args)

            for k, v in metric_dict:
                self.log.info(f"metrics - {data_type} - {metric_cfg.name} - {k} - {v}")
