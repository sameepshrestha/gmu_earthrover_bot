from mmseg.registry import DATASETS
from mmseg.datasets import CityscapesDataset


@DATASETS.register_module()
class MapillaryVistasDataset(CityscapesDataset):
    """MapillaryVistasDataset dataset."""

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
