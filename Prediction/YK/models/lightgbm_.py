#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author    :   yujia.Ke
# Time      :   2022/12/1 19:52
# File      :   lightgbm_.py
# Objective :   LGB wrapper.

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer

import src.utils as utils
from src.model.output import OutputModel

logger = logging.getLogger()


class LightGBMModel(object):
    """LightGBM Model."""

    def __init__(self, input_data, pred_date: str, horizon: int, config: dict):
        """
        num_leaves: 31  # Maximum tree leaves for base learners
        max_depth: -1  # Maximum tree depth for base learners, <=0 means no limit
        learning_rate: 0.1  # Boosting learning rate
        # n_estimators: 100  # Number of boosted trees to fit
        subsample: 1  # 建树的样本采样比例
        colsample_bytree: 1  # 建树的特征选择比例
        num_round: 1000  # number of boosting iterations
        """
        self.data = input_data.copy()

        self.predDate = pred_date
        self.predDuration = horizon

        self.default_param = config

        self.num_leaves = None
        self.max_depth = None
        self.learning_rate = None  # self.eta
        # self.n_estimators = None
        self.subsample = None
        self.colsample_bytree = None
        self.num_round = None

        logger.info("Model initialized...")

    def prepare_data_parameters(self):
        """
        prepare lightgbm parameters
        """
        # default parameters
        model_param = {"num_leaves": self.default_param["num_leaves"],
                       'max_depth': self.default_param['max_depth'],
                       'learning_rate': self.default_param['learning_rate'],
                       # 'n_estimators': self.default_param['n_estimators'],
                       'subsample': self.default_param['subsample'],
                       'colsample_bytree': self.default_param['colsample_bytree'],
                       'num_round': self.default_param['num_round'],
                       }

        # input parameters
        self.num_leaves = model_param["num_leaves"]
        self.max_depth = model_param['max_depth']
        self.learning_rate = model_param['learning_rate']
        # self.n_estimators = model_param['n_estimators']
        self.subsample = model_param['subsample']
        self.colsample_bytree = model_param['colsample_bytree']
        self.num_round = model_param['num_round']

        logger.info("Parameters ready...")

    def fit_predict(self):
        """
        fit & predict
        get pred & hist
        """
        self.prepare_data_parameters()

        if self.data.empty:
            logger.info("No history data!")
            raise utils.NoTrainingDataException()

        # 调用LightGBM模型，使用训练集数据进行训练（拟合）
        logger.info('num_leaves:{}'.format(int(self.num_leaves)))
        logger.info('max_depth:{}'.format(int(self.max_depth)))
        logger.info('learning_rate:{}'.format(self.learning_rate))
        logger.info('num_iterations:{}'.format(int(self.num_round)))
        logger.info('subsample:{}'.format(self.subsample))
        logger.info('colsample_bytree:{}'.format(self.colsample_bytree))

        num_leaves = int(self.num_leaves) if int(self.num_leaves) <= 14 else 14
        max_depth = int(self.max_depth) if int(self.max_depth) <= 4 else 4
        subsample = int(self.subsample) if int(self.subsample) <= 0.8 else 0.8

        # 增加若干特征
        self.data["month"] = self.data.t.dt.month
        self.data["month"] = self.data.t.dt.day.astype(int)
        self.data["weekday"] = (self.data.t.dt.weekday + 1).astype(int)
        self.data["workday"] = (self.data.t.dt.weekday <= 4).astype(int)
        # 保证后续对时间的筛选不报错
        self.data['t'] = self.data.t.astype(str)

        res_list = list()
        for company, data in self.data.groupby("id"):
            train = data[data.t < self.predDate].copy()
            test = data[data.t >= self.predDate].copy()

            # determine max pred duration
            # do not predict if all features are nan
            max_test_date = max(test.t)
            start_date = datetime.strftime(datetime.strptime(self.predDate, '%Y-%m-%d') - timedelta(1), '%Y-%m-%d')
            date1 = datetime.strptime(start_date, '%Y-%m-%d')
            date2 = date1 + timedelta(self.predDuration)
            max_duration_date = str(date2)[:10]
            max_date = min(max_test_date, max_duration_date)
            test = test[test.t <= max_date].copy()

            # 所有可用特征列
            feat_col = list(set(list(train.columns)) - {"id", 't', 'y'})
            if train.shape[0] <= 0:
                logger.info("No training data")
                raise utils.NoTrainingDataException()
            elif test.shape[0] <= 0 or len(feat_col) <= 0:
                logger.info("no feature")
                raise utils.NoFeatureException()
            else:
                x_train = train[feat_col].copy()
                y_train = train['y'].copy()
                x_test = test[feat_col].copy()
                y_test = test['y'].copy()
                # # 空值处理，默认方法：使用特征列的平均值进行填充
                # my_imputer = SimpleImputer()
                # x_train = my_imputer.fit_transform(x_train)
                # x_test = my_imputer.transform(x_test)
                # # 转换为Dataset数据格式
                # lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=["weekday", "workday", ])
                # lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, categorical_feature=["weekday", "workday", ])

                # Mean forecast
                my_model = lgb.LGBMRegressor(objective="regression",
                                             num_leaves=num_leaves,
                                             max_depth=max_depth,
                                             learning_rate=self.learning_rate,
                                             # n_estimators=int(self.n_estimators),
                                             subsample=subsample,
                                             colsample_bytree=0.6,
                                             num_iterations=int(self.num_round),
                                             verbosity=-1,
                                             boosting_type='gbdt',
                                             min_child_samples=15,
                                             bagging_fraction=0.6,
                                             feature_fraction=0.6,
                                             max_bin=225,
                                             num_threads=150,
                                             lambda_l1=0.6,
                                             lambda_l2=0,
                                             min_split_gain=0.1,
                                             )
                logger.info("model already")
                model_start = time.time()
                my_model.fit(x_train, y_train,
                             verbose=False,
                             eval_set=[(x_test, y_test)],
                             eval_metric="rmse",
                             early_stopping_rounds=5,
                             )
                model_end = time.time()
                logger.info("Model fit cost: {} seconds".format(str(round((model_end - model_start), 2))))
                logger.info("fit already")

                # 使用模型对测试集数据进行预测
                test['yhat'] = my_model.predict(x_test)
                logger.info("predict already")
                pred = test[['id', 't', 'yhat']].copy()

                # interval forecast
                for alpha in (.95, .05):
                    my_model = lgb.LGBMRegressor(objective="quantile",
                                                 num_leaves=num_leaves,
                                                 max_depth=max_depth,
                                                 learning_rate=self.learning_rate,
                                                 # n_estimators=int(self.n_estimators),
                                                 subsample=subsample,
                                                 colsample_bytree=0.6,
                                                 num_iterations=int(self.num_round),
                                                 verbosity=-1,
                                                 boosting_type='gbdt',
                                                 min_child_samples=15,
                                                 bagging_fraction=0.6,
                                                 feature_fraction=0.6,
                                                 max_bin=225,
                                                 num_threads=150,
                                                 lambda_l1=0.6,
                                                 lambda_l2=0,
                                                 min_split_gain=0.1,
                                                 alpha=alpha,
                                                 )
                    logger.info("model already")
                    model_start = time.time()
                    my_model.fit(x_train, y_train,
                                 verbose=False,
                                 eval_set=[(x_test, y_test)],
                                 eval_metric="rmse",
                                 early_stopping_rounds=5)
                    model_end = time.time()
                    logger.info("Model fit cost: {} seconds".format(str(round((model_end - model_start), 2))))
                    logger.info("fit already")

                    # 使用模型对测试集数据进行预测
                    test[f"yhat_{alpha}"] = my_model.predict(x_test)
                    logger.info("predict already")
                    pred = pd.merge(left=pred, right=test[['id', 't', f"yhat_{alpha}"]], on=["id", 't'])

                res_list.append(pred)

        res = pd.concat(res_list)
        res.t = pd.to_datetime(res.t)
        logger.info("Results generated")
        return res


if __name__ == "__main__":
    pass
