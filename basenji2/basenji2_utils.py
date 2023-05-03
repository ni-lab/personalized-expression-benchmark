import numpy as np

ENFORMER_SEQ_LENGTH = 393216  # we trim enformer consensus sequences to make our predictions


def avg_center_bins(preds: np.ndarray, n_center: int = 10) -> float:
    """
    Average center bins symmetrically, assuming number of bins in preds is even.
    """
    assert preds.ndim == 2, f"preds should have 2 dimensions, not {preds.ndim}"
    mid = preds.shape[0] // 2
    start = mid - n_center // 2
    end = mid + n_center // 2
    return np.mean(preds[start:end], axis=0)  # get the mean of the center bins

