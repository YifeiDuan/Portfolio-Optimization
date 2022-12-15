import datetime
import json
import os
import re
import threading
import time
from functools import wraps
from os import walk

import pandas as pd


def drop_prefix(df):
    """去掉字段的前缀（表名）"""
    df.rename(lambda x: x.split('.')[1], axis='columns', inplace=True)
    return df


def cal_tm_diff(end, start):
    """计算时长：小时"""
    end_tm = to_time_type(end)
    start_tm = to_time_type(start)
    if end_tm and start_tm:
        time_dif = (end_tm - start_tm).total_seconds() / 3600
    else:
        time_dif = None

    return time_dif


class Timer(object):
    """计时器类"""

    def __init__(self, msg='', logger=None, verbose=True):
        self.verbose = verbose
        self.msg = msg
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        # self.msecs = self.secs * 1000  # millisecs
        if self.verbose and self.logger:
            self.logger.info('%s cost time: %.3f s' % (self.msg, self.secs))


def Singleton():
    """
    单例模式装饰器

    :return:
    """
    # 闭包绑定线程锁
    lock = threading.Lock()

    def decorator(cls):
        # 替换 __new__ 函数
        instance_attr = '_instance'
        # 获取原来的__new__函数 防止无限递归
        __origin_new__ = cls.__new__

        @wraps(__origin_new__)
        def __new__(cls_1, *args, **kwargs):
            if not hasattr(cls_1, instance_attr):
                with lock:
                    if not hasattr(cls_1, instance_attr):
                        setattr(cls_1, instance_attr, __origin_new__(cls_1, *args, **kwargs))
            return getattr(cls_1, instance_attr)

        cls.__new__ = __new__

        # 替换 __init__函数 原理同上
        init_flag = '_init_flag'
        __origin_init__ = cls.__init__

        @wraps(__origin_init__)
        def __init__(self, *args, **kwargs):
            if not hasattr(self, init_flag):
                with lock:
                    if not hasattr(self, init_flag):
                        __origin_init__(self, *args, **kwargs)
                        setattr(self, init_flag, True)

        cls.__init__ = __init__
        return cls

    return decorator


def judge_time_format(tm_str):
    """
    时间格式校验

    :param tm_str: the string type of the time
    :return: formatted time str
    """
    if tm_str is not None:
        tm = to_time_type(tm_str)
        if tm is None:
            return ''
        else:
            return tm.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return tm_str


def to_time_type(tm_str):
    """
    字符串转时间类型

    :param tm_str:
    :return:
    """
    pattern1 = r'\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}'
    pattern2 = r'\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{1,2}'
    pattern3 = r'\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{1,2}'
    pattern4 = r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}'
    pattern5 = r'\d{4}-\d{1,2}-\d{1,2}'
    pattern6 = r'\d{4}-\d{1,2}'

    if re.match(pattern1, tm_str):
        tm = datetime.datetime.strptime(re.match(pattern1, tm_str).group(), '%Y-%m-%d %H:%M')
    elif re.match(pattern2, tm_str):
        tm = datetime.datetime.strptime(re.match(pattern2, tm_str).group(), '%Y/%m/%d %H:%M')
    elif re.match(pattern3, tm_str):
        tm = datetime.datetime.strptime(re.match(pattern3, tm_str).group(), '%m-%d-%Y %H:%M')
    elif re.match(pattern4, tm_str):
        tm = datetime.datetime.strptime(re.match(pattern4, tm_str).group(), '%m/%d/%Y %H:%M')
    elif re.match(pattern5, tm_str):
        tm = datetime.datetime.strptime(re.match(pattern5, tm_str).group(), '%Y-%m-%d')
    elif re.match(pattern6, tm_str):
        tm = datetime.datetime.strptime(re.match(pattern6, tm_str).group(), '%Y-%m')
    else:
        tm = None
    return tm


def MSG(msg):
    """添加前缀到日志内容"""
    return '|%(taskId)d|%(jobId)s|%(identifier)s| %(logMsg)s' % {
        "taskId": global_dict.get_value("taskId"),
        "jobId": str(global_dict.get_value("jobId")),
        "identifier": str(global_dict.get_value("logMsg")),
        "logMsg": msg}


class GlobalDict:
    """global variables set"""

    def __init__(self):
        self._global_dict = {}

    def set_value(self, key, value):
        self._global_dict[key] = value

    def get_value(self, key, def_value=None):
        try:
            return self._global_dict[key]
        except KeyError:
            return def_value

    def empty_dict(self):
        """清空"""
        for key in self._global_dict.keys():
            self._global_dict[key] = None


global_dict = GlobalDict()  # 全局变量


def timeindex_resample(df, col, interval):
    """
    升采样，用空值填充

    :param df:
    :param col:
    :param interval:
    :return:
    """
    df.set_index(col, inplace=True)
    df = df.resample(interval).asfreq()  # asfreq 用空值填充
    return df


def get_current_timestr(time_format):
    """
    get current time string

    :param time_format: time format
    :return:
    """
    return datetime.datetime.now().strftime(time_format)


def time_to_str(time_format, tm):
    """
    change time type to string type

    :param time_format:
    :param tm:
    :return:
    """
    return tm.strftime(time_format)


def df_to_json(df, orient_type='records'):
    """
    change dataframe to json

    :param df:
    :param orient_type: to_json::orient
    :return:
    """
    if df is not None:
        rs = json.loads(df.to_json(orient=orient_type))
    else:
        rs = None
    return rs


def read_dataframe_list(file_path):
    """
    read dataframe from dir and append into one list

    :param file_path: the path of dir
    :param df_list: the list of dataframe
    :return:
    """
    df_list = []
    for dir_path, dirs, file_names in walk(file_path):
        for file in file_names:
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(file_path)
            df_list.append(df)

    return df_list


def bins_cut(df, bins, labels, split_col):
    """
    # sample
    bins = [-1, 500, 1000, 5000, float('inf')]
    labels = ['0-500', '500-1000', '1000-5000', '>5000']
    df['mean_level'] = bins_cut(df, bins, labels, 'mean')

    :param df:
    :param bins:
    :param labels:
    :param split_col:
    :return:
    """
    return pd.cut(df[split_col], bins, right=True, labels=labels)


class MyException(Exception):

    def __init__(self, *args):
        self.args = args


class NoRecentDataException(MyException):

    def __init__(self, message=None, code=401):
        self.args = (code, "no recent data", message)
        self.message = message
        self.code = code


class NoEnoughDataException(MyException):

    def __init__(self, message=None, code=402):
        self.args = (code, 'no enough data', message)
        self.message = message
        self.code = code


class NoTrainingDataException(MyException):

    def __init__(self, message=None, code=501):
        self.args = (code, 'no training data', message)
        self.message = message
        self.code = code


class NoFeatureException(MyException):

    def __init__(self, message=None, code=502):
        self.args = (code, 'no feature', message)
        self.message = message
        self.code = code
