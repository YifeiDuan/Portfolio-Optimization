#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time      : 2022/12/4 15:30
# @Author    : keyuj
# @File      : metric.py
# @Project   : Project
# @Objective :

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def mape(raw, result):
    """
    Mean absolute percentage error.

    :param raw: id, t, y
    :param result: id, t, yhat
    :return:
    """
    data = pd.merge(left=raw, right=result, on=["id", 't'], how="inner")
    data["mape"] = np.absolute((data.y - data.yhat) / data.y)
    # print("Scikit-learn MAPE:", mean_absolute_percentage_error(data.y, data.yhat))

    return data


def smape(raw, result):
    """
    Symmetric mean absolute percentage error.

    :param raw:
    :param result:
    :return:
    """
    data = pd.merge(left=raw, right=result, on=["id", 't'], how="inner")
    data["smape"] = 2 * np.absolute((data.y - data.yhat) / (data.y + data.yhat))

    return data


def wmape(raw, result):
    """
    Weighted mean absolute percentage error.

    :param raw:
    :param result:
    :return:
    """
    data = pd.merge(left=raw, right=result, on=["id", 't'], how="inner")

    return np.sum(np.absolute((data.y - data.yhat) / data.y * data.y)) / np.sum(np.absolute(data.y))


def r2(raw, result_bsl, result, built_in=True):
    """
    1 - Residual sum of squared / total sum of squared

    :param raw:
    :param result_bsl:
    :param result:
    :param built_in:
    :return:
    """
    data = pd.merge(left=raw, right=result_bsl, on=["id", 't'], how="inner")
    data = pd.merge(left=data, right=result, on=["id", 't'], how="inner")

    if built_in:
        return r2_score(y_true=data.y, y_pred=data.yhat)
    else:
        return 1 - np.sum(np.square(data.yhat - data.y)) / np.sum(np.square(data.ybar - data.y))
