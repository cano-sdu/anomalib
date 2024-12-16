import logging

import torch

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

from .base import BaseThreshold

logger = logging.getLogger(__name__)


class HighRecallThreshold(BinaryPrecisionRecallCurve, BaseThreshold):
    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

    def compute(self) -> torch.Tensor:
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        if not any(1 in batch for batch in self.target):
            msg = (
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
            logging.warning(msg)

        precision, recall, thresholds = super().compute()

        idx_reversed = torch.argmax(torch.flip(recall, [0]))  # Find last occurrence of the highest recall
        idx = recall.size()[0] - idx_reversed - 1

        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[idx]

    def __repr__(self):
        """Return threshold value within the string representation.

        Returns:
            str: String representation of the class.
        """
        return f"{super().__repr__()} (value={self.value:.2f})"