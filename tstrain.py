# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:51:09 2017

@author: Lunch 
Timeseries
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#timestamp period interval
rng = pd.date_range('2017-11-30',periods=10,freq='B')
time = pd.Series(np.random.randn(10),index=pd.date_range(dt.datetime(2017,11,1),periods=10,freq='B'))
time.truncate(before='20171103')#那部分不要了
#时间对照表，很管用

#时间戳
pd.Timestamp('20160711 10:13:33')
#时间区间
pd.Period('2016-01-01')

#Time offsets

pd.Timedelta(days=1)
dt.timedelta(days=1)

pd.Period('20180202 10:10') + pd.Timedelta(days=1)

p1 = pd.period_range('20161001',freq='B',periods=360)
p2 = pd.period_range('20161001',freq='1D1H',periods=360)

periods = [pd.Period('2016-01'), pd.Period('2016-02'), pd.Period('2016-03')]
ts = pd.Series(np.random.randn(len(periods)), index = periods)

#period 索引和 difference
ts = pd.Series(range(10), pd.date_range('07-10-16 8:00', periods = 10, freq = 'H'))
ts_period=ts.to_period()
ts['2016-07-10 8:30':'2016-07-10 11:45'] 

#resampling 
rng = pd.date_range('20170101',periods=90,freq='D')
ts = pd.Series(np.random.randn(len(rng)),index= rng)

ts.resample('M').sum()
ts.resample('3D').apply(lambda x: x.std())

day3Ts = ts.resample('3D').mean()
day3Ts.resample('D').interpolate('time')

