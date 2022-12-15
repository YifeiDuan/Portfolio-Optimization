import logging
import numpy as np
from datetime import datetime, timedelta

import pandas as pd

import src.utils as utils
from src.model.prophet_forecaster import Prophet


class MITProphet(object):
    """
    start_date: predDate - 1
    min_last_obs: max # last null observation pts allowed to use the model
    min_obs: # history observation pts required to use the model
    n_changepoints: # potential changepoints to include
    changepoint_range: proportion of history in which trend changepoints will
        be estimated   
    seasonality_mode: 'additive' (default) or 'multiplicative'
    confidence: width of the uncertainty intervals provided for the forecast   
    algorithm: Newton or LBFGS
    periods: # periods to predict ahead
    """

    def __init__(self, input_data, pred_date: str, horizon: int, config: dict):

        self.input_data = self.data = input_data
        self.holidays = None

        self.predDate = pred_date
        self.predDuration = horizon

        self.default_param = config

        self.min_last_obs = None
        self.min_obs = None
        self.n_changepoints = None
        self.changepoint_range = None
        self.seasonality_mode = None
        self.confidence = None

        self.periods = None
        self.algorithm = None

        logging.info("Model initialized...")

    def prepare_data_parameters(self):
        """
        prepare data and parameters for model
       """
        # prepare model parameters
        model_param = {'min_last_obs': self.default_param['min_last_obs'],
                       'change_algorithm_obs': self.default_param['change_algorithm_obs'],
                       'min_obs': self.default_param['min_obs'],
                       'max_changepoints': self.default_param['max_changepoints'],
                       'changepoint_range': self.data.shape[0] * 0.2,
                       'seasonality_mode': self.default_param['seasonality_mode'],
                       'confidence': self.default_param['confidence']}

        if model_param["seasonality_mode"] == 1:
            self.seasonality_mode = "additive"
        else:
            self.seasonality_mode = "multiplicative"
        if self.data.shape[0] <= model_param["change_algorithm_obs"]:
            self.algorithm = "Newton"
        else:
            self.algorithm = "LBFGS"

        self.changepoint_range = 1 - model_param['changepoint_range'] / self.data.shape[0]
        self.n_changepoints = int(model_param['max_changepoints'])
        self.min_last_obs = int(model_param['min_last_obs'])
        self.min_obs = int(model_param['min_obs'])
        self.confidence = model_param['confidence']

        # check data
        self.data = self.data[self.data.t < pd.to_datetime(self.predDate)]
        recent_allowed_date = datetime.strptime(self.predDate, "%Y-%m-%d") - timedelta(self.min_last_obs)
        for company, data in self.data.groupby("id"):
            # 检查是否有数据
            if data.empty:
                raise utils.NoEnoughDataException(company)
            # 检查数据量是否足够
            if max(data.t) < recent_allowed_date:
                logging.error(company + ": no recent data => cannot predict by MIT Prophet")
                raise utils.NoRecentDataException(company)
            elif data.shape[0] < self.min_obs:
                logging.info(company + ": no enough data => cannot predict by MIT Prophet")
                raise utils.NoEnoughDataException(company)

        # prepare periods ahead
        dt1 = datetime.strptime(self.predDate, "%Y-%m-%d") + timedelta(int(self.predDuration) - 1)
        dt2 = datetime.strptime(str(max(self.data.t))[0:10], "%Y-%m-%d")
        self.periods = max(0, (dt1 - dt2).days)

        logging.info("Data & parameters ready...")

    def get_holidays(self):
        """
        get area for holiday based on input data
        get holiday table based on taskId and area from mysql then reformat
        """
        # get area name
        id_ = self.data.id[0]
        area = id_[:3]
        # area_name = ''

        if area == '852':
            area_name = 'hongkong'
        elif area == '853':
            area_name = 'macao'
        elif area == '886':
            area_name = 'taiwan'
        else:
            area_name = 'mainland'

        holidays = pd.read_csv("data\holiday_1.csv")
        holidays = holidays[holidays['area'] == 'mainland']
        holidays = holidays[['holiday', 'ds', 'lower_window', 'upper_window', 'additive']]

        # format holiday table
        if holidays.empty or holidays is None:
            holidays = None
            logging.warning("Empty holiday table...")
        else:
            holidays.holiday = holidays.holiday.astype(str)
            holidays.ds = holidays.ds.astype(str)
            holidays.lower_window = holidays.lower_window.astype(int)
            holidays.upper_window = holidays.upper_window.astype(int)
            holidays.loc[holidays.additive == 1, "additive"] = True
            holidays.loc[holidays.additive == 0, "additive"] = False

        self.holidays = holidays
        logging.info("Holiday table ready")

    def fit_predict(self, additional_regressor=True):
        """
        fit and predict the model
        get pred, hist, pred_confidence, pred_component
        """
        self.prepare_data_parameters()
        # self.get_holidays()

        results = list()
        for company, data in self.data.groupby("id"):
            m = Prophet(
                # holidays=self.holidays,
                yearly_seasonality=True,
                monthly_seasonality=True,
                weekly_seasonality=True,
                n_changepoints=self.n_changepoints,
                changepoint_range=self.changepoint_range,
                seasonality_mode=self.seasonality_mode,
                interval_width=self.confidence)
            if company == "BABA":
                m.add_country_holidays(country_name="China")
            else:
                m.add_country_holidays(country_name="US")
            dt = data[['t', 'y']].rename(columns={'t': "ds"})
            # regressor: workday or not, day of week
            if additional_regressor:
                data["weekday"] = data.t.dt.weekday + 1
                data["workday"] = np.asarray(data.t.dt.weekday <= 4, dtype=np.int)
                # dt = data[['t', 'y', "last_date_price"]].rename(columns={'t': "ds"})
                # m.add_regressor("last_date_price")
                dt = data[['t', 'y', "weekday"]].rename(columns={'t': "ds"})
                m.add_regressor("weekday")
            m.fit(df=dt, algorithm=self.algorithm)
            future = m.make_future_dataframe(periods=self.periods, include_history=True)
            if additional_regressor:
                # future["last_date_price"] = self.input_data[(self.input_data.id == company) & (
                #     self.input_data.t.isin(future.ds.tolist()))].last_date_price.tolist()
                # future["workday"] = np.asarray(future.ds.dt.weekday <= 4, dtype=np.int)
                future["weekday"] = future.ds.dt.weekday + 1
            # # 考虑只对设定日期之后的进行预测
            # future = future[future.ds >= self.predDate].reset_index(drop=True)
            forecast = m.predict(future)

            # # history
            # hist = self.data.copy()
            # hist.reset_index(drop=True, inplace=True)
            # hist = hist[hist.t < self.predDate]

            # prediction & confidence
            pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={"ds": 't'})
            pred["id"] = company
            results.append(pred[["id", 't', "yhat", "yhat_lower", "yhat_upper"]])
            logging.info(company + ": results generated!")

        return pd.concat(results)

    def plot(self):
        pass
