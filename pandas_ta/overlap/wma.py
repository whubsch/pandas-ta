# -*- coding: utf-8 -*-
from pandas import Series
from pandas_ta import Imports
from pandas_ta.utils import get_offset, verify_series
import numpy as np


def wma(close, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Weighted Moving Average (WMA)"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    asc = asc if asc else True
    offset = get_offset(offset)

    # Calculate Result
    if Imports["talib"]:
        from talib import WMA
        wma = WMA(close, length)
    else:
        total_weight = 0.5 * length * (length + 1)
        weights_ = np.arange(1, length + 1)
        weights = weights_ if asc else np.flip(weights_)

        def _linear(x):
            return np.dot(x, weights) / total_weight

        values = [
            _linear(each)
            for each in np.lib.stride_tricks.sliding_window_view(np.array(close), length)
        ]
        wma_ds = Series([np.NaN] * (length - 1) + values)
        wma_ds.index = close.index

    # Offset
    if offset != 0:
        wma_ds = wma_ds.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        wma_ds.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        wma_ds.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    wma_ds.name = f"WMA_{length}"
    wma_ds.category = "overlap"

    return wma_ds


wma.__doc__ = \
"""Weighted Moving Average (WMA)

The Weighted Moving Average where the weights are linearly increasing and
the most recent data has the heaviest weight.

Sources:
    https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

Calculation:
    Default Inputs:
        length=10, asc=True
    total_weight = 0.5 * length * (length + 1)
    weights_ = [1, 2, ..., length + 1]  # Ascending
    weights = weights if asc else weights[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    WMA = close.rolling(length)_.apply(linear_weights(weights), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    asc (bool): Recent values weigh more. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
