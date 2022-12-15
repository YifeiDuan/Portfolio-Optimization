#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time      : 2022/12/4 14:33
# @Author    : keyuj
# @File      : preprocess.py
# @Project   : Project
# @Objective :

import arrow
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

# # 显示所有列
# pd.set_option('display.max_columns', None)
# # 显示所有行
# pd.set_option('display.max_rows', None)
# # 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 100)


def read_data(path_=r"E:\MIT\Courses\6.7200[J]15.093 Optimization Methods\Project\data\stock_prices.csv",
              max_length=36):
    """
    Read in raw data and melt into usable columns.

    :param path_: file path
    :param max_length: in months
    :return:
    """
    data_org = pd.read_csv(filepath_or_buffer=path_, header=0, index_col=False)
    # column to row (reversed pivot)
    data = data_org.melt(id_vars="date", value_vars=None, var_name="company", value_name="stock_price").dropna(axis=0)
    # only use recent data
    max_date = arrow.get(data["date"].max()).shift(months=- max_length).datetime.strftime("%Y-%m-%d")
    data = data[data["date"] >= max_date].dropna(axis=0)
    # calculate historical return
    data = data.sort_values(by=["company", "date"]).reset_index(drop=True)
    return_list, last_date_price_list = list(), list()
    for company, c_data in data.groupby("company"):
        return_list += c_data.stock_price.pct_change().tolist()
        last_date_price_list += c_data.stock_price.shift(1).tolist()
    data["return"], data["last_date_price"] = return_list, last_date_price_list
    # rename
    data.rename(columns={"date": 't', "company": "id", "return": 'y'}, inplace=True)
    # assert date type
    data.id = data.id.astype(str)
    data.t = pd.to_datetime(data.t.astype(str), format="%Y-%m-%d")
    data.y = data.y.astype(float)
    # sort
    data = data.dropna(axis=0).sort_values(by=["id", 't']).reset_index(drop=True)

    # # y * 100
    # data.y *= 100

    logging.info(f"Shape of available data (drop Nan): {data.shape}")
    logging.info(f"Data sample:\n{data.head()}")
    return data


def read_config(model_name="prophet"):
    """
    Read in model config parameters.

    :param model_name:
    :return:
    """
    with open(os.path.join(r"E:\MIT\Courses\6.7200[J]15.093 Optimization Methods\Project\src\model",
                           f"{model_name}.yaml"), "rt", encoding="U8") as f:
        config = yaml.safe_load(f)
    logging.info(f"Config dict:\n{config}")

    return config


def plot_data(data, company: str = "AAPL"):
    """
    Time series data plot.

    :param data: columns [t, id, stock_price, y, last_date_price]
    :param company:
    :return:
    """
    # plt.clf()
    # # 设置坐标轴名称
    # plt.xlabel("production")
    # plt.ylabel("price")
    # plt.title("Supply Curve with Doubled Electricity Cost")
    # plt.plot(x, y)
    # plt.plot(x_1, y_1)
    # plt.legend(["Original", "Double Electricity Cost"])
    # plt.show()

    data = data[data.id == company].copy()
    x, stock_price, historic_return = data.t, data.stock_price, data.y

    plt.clf()
    _, ax = plt.subplots(1, 1)
    # 共享x轴，生成次坐标轴
    ax_sub = ax.twinx()
    # 设置坐标轴名称
    ax.set_xlabel("time series")
    ax.set_ylabel("price ($)")
    ax_sub.set_ylabel("return (%)")
    ax.set_title(f"Time series of stock price and return of {company}")
    l1 = ax.plot(x, stock_price, color="gray", label="stock price ($)")
    l2 = ax_sub.plot(x, historic_return, "r-", label="historic return (%)")
    # plt.legend(handles=[l1, l2], labels=["stock price ($)", "historic return (%)"], loc=0)
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    read_data()
    read_config()
