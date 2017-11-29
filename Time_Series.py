# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:02:38 2017
time series
@author: Lunch
"""

from __future__ import absolute_import, division, print_function
import sys
import os
import pandas as pd
import numpy as np

# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

import matplotlib.pylab as plt
import seaborn as sns

#pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
#np.set_printoptions(precision=5, suppress=True) # numpy
#
#pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster') 

#Read the data
Sentiment = 'F:\\test_data\\sentiment.csv'
Sentiment = pd.read_csv(Sentiment, index_col=0, parse_dates=[0])

sentiment_short = Sentiment.loc['2005':'2016']
#数据图例查看
sentiment_short.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25,0.5))
plt.title("Consumer Sentiment")
sns.despine()
'''着手开始进行 ARIMA中 D的计算'''
sentiment_short['diff_1'] = sentiment_short.UMCSENT.diff(1)
sentiment_short['diff_2'] = sentiment_short.diff_1.diff(1)
#画图观察波动性是否大
sentiment_short.plot(subplots=True, figsize=(18,12))

del sentiment_short['diff_2']
del sentiment_short['diff_1']

#画图算ACF PACF 
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sentiment_short, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2=fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sentiment_short,lags=20,ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()

#散点图表示

lags=9

ncols=3
nrows=int(np.ceil(lags/ncols))

fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))

for ax, lag in zip(axes.flat, np.arange(1,lags+1, 1)):
    lag_str = 't-{}'.format(lag)
    X = (pd.concat([sentiment_short, sentiment_short.shift(-lag)], axis=1,
                   keys=['y'] + [lag_str]).dropna())

    X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
    corr = X.corr().as_matrix()[0][1]
    ax.set_ylabel('Original')
    ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
    ax.set_aspect('equal');
    sns.despine();

fig.tight_layout();

# 只管的图表
def tsplot(y,lags=None, title='', figsize=(14,8)):
    fig = plt.figure(figsize=figsize)
    layout=(2,2)
    ts_ax = plt.subplot2grid(layout, (0,0))
    hist_ax = plt.subplot2grid(layout,(0,1))
    acf_ax = plt.subplot2grid(layout,(1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    
    y.plot(ax = ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax,kind='hist',bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags = lags ,ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

tsplot(sentiment_short, title='Consumer Sentiment',lags=36)







