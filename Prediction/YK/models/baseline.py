#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time      : 2022/12/4 21:25
# @Author    : keyuj
# @File      : baseline.py
# @Project   : Project
# @Objective :

from itertools import product

import pandas as pd


def moving_average(data, pred_date: str, horizon: int = 1):
    """

    :param data:
    :param pred_date:
    :param horizon:
    :return:
    """
    future = data[["id", 'y']].groupby("id").mean().reset_index()
    t_list = pd.date_range(start=pred_date, periods=horizon)
    id_list = future.id.tolist()
    t_id = pd.DataFrame(product(t_list, id_list), columns=['t', "id"])
    future = pd.merge(left=t_id, right=future).rename(columns={'y': "ybar"}).reset_index(drop=True)

    return future
