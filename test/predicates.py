import numpy as np


def within_bounds(
    arr: np.array,
    lo: float,
    hi: float,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True
):
    assert lo <= hi, "lower bound must be less than or equal to higher bound"

    lo_arr = (arr >= lo) if lo_inclusive else (arr > lo)
    hi_arr = (arr <= hi) if hi_inclusive else (arr < hi)
    return np.all(lo_arr & hi_arr)
