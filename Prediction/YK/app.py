#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time      : 2022/12/4 14:31
# @Author    : Yujia Ke
# @File      : app.py
# @Project   : Project
# @Objective :

import logging

import preprocess
import model.baseline as bsl
import model.prophet as prophet
import model.lightgbm_ as lgb
import model.metric as metric

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 读入
data = preprocess.read_data(max_length=36)
config_prophet = preprocess.read_config("prophet")
config_lgb = preprocess.read_config("lightgbm")
print(set(data.id))

# # 绘图
# preprocess.plot_data(data, company="GOOG")

# 预测日期及外推其
# 注意只有工作日有数据
pred_date, horizon = "2018-03-01", 40
res_bsl = bsl.moving_average(data, pred_date, horizon)

# Prophet
model_prophet = prophet.MITProphet(data, pred_date, horizon=horizon, config=config_prophet)
res_prophet = model_prophet.fit_predict()
eva_mape = metric.mape(raw=data, result=res_prophet)
eva_smape = metric.smape(raw=data, result=res_prophet)
print(eva_mape[["id", "mape"]].dropna(axis=0).groupby("id").mean())
print(eva_smape[["id", "smape"]].dropna(axis=0).groupby("id").mean())
logger.info(f"Weighted MAPE: {metric.wmape(raw=data, result=res_prophet)}")
logger.info(f"R squared: {metric.r2(raw=data, result_bsl=res_bsl, result=res_prophet)}")

# Lightgbm
model_lgb = lgb.LightGBMModel(input_data=data, pred_date=pred_date, horizon=horizon, config=config_lgb)
res_lgb = model_lgb.fit_predict()
eva_mape_lgb = metric.mape(raw=data, result=res_lgb)
eva_smape_lgb = metric.smape(raw=data, result=res_lgb)
print(eva_mape_lgb[["id", "mape"]].dropna(axis=0).groupby("id").mean())
print(eva_smape_lgb[["id", "smape"]].dropna(axis=0).groupby("id").mean())
logger.info(f"Weighted MAPE: {metric.wmape(raw=data, result=res_lgb)}")
logger.info(f"R squared: {metric.r2(raw=data, result_bsl=res_bsl, result=res_lgb)}")

if __name__ == "__main__":
    pass
