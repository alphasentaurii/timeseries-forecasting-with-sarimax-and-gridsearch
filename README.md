# Time Series Forecasting with SARIMAX and Gridsearch

![GitHub repo size](https://img.shields.io/github/repo-size/hakkeray/timeseries-forecasting-with-sarimax-and-gridsearch)
![GitHub license](https://img.shields.io/github/license/hakkeray/timeseries-forecasting-with-sarimax-and-gridsearch?color=black)

Time Series Forecasting with SARIMAX and Gridsearch is a `housing market prediction model` that uses `seasonal ARIMA time-series analysis and GridsearchCV` to `recommend the top 5 zip codes for purchasing a single-family home in Westchester, New York`.

![](./10605.png)

## Summary

Prior to training, I set out to identify trends, seasonality, and autoregression elements within the Zillow Housing Market Dataset. I then use a fitting procedure to find the coefficients of a regression model, including several plots and statistical tests of the residual errors. The top 5 zip code recommendations rely on the following factors: `highest ROI`, `lowest confidence intervals`, and `shortest commute time` from Grand Central Station.

## Highlights

Along with several custom time series analysis helper functions I wrote for this project, I also extrapolate the USZIPCODE pypi library to account for several exogenous factors, including average income levels, using an updated version of the `zipstats()` function I wrote for a previous data science project [Predicting Home Values with Multiple Linear Regression](https://github.com/hakkeray/predicting-home-values-with-multiple-linear-regression).

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a `<Windows/Linux/Mac>` machine.
* You have installed the latest version of `Jupyter Notebook`
* Install PyPi and FSDS libraries:

```bash
!pip install -U uszipcode
!pip install -U fsds_100719
```

## Running the Time Series Forecasting with SARIMAX and Gridsearch Project

To run this project locally, follow these steps:

In the command line/terminal:

```bash
$ git clone https://github.com/hakkeray/timeseries-forecasting-with-sarimax-and-gridsearch
$ cd timeseries-forecasting-with-sarimax-and-gridsearch
$ jupyter notebook
```

Please note that the Zillow data set contains *millions* of US Zipcodes. If you want to fork this project and follow along with some of the steps I took, you can apply the same model I did to a completely different county. The only part that you'll need to skip (or adjust) is the Metro North railroad section, since this only applies to Westchester County, New York. 

## Business Case

**Goal**

Recommend top 5 zipcodes for client interested in buying a single-family home in Westchester County, NY within the next two years.

**Required Parameters (CLIENT)**

1. Timeframe: Purchase of new home would take place within the next two years
2. Budget: Cost to buy not to exceed `800,000 USD` maximum
3. Location: Zip Code search radius restricted to Westchester County, New York

**Ideal Parameters (CLIENT)**

* Commute: Towns/Villages having shortest/fastest commute New York City
* School District: Zip codes include towns with A/A+ school district rating

**Success Criteria for Model**
1. Maximum ROI (highest forecasted home value increase for lowest upfront cost)
2. Confidence Intervals
3. Exogenous factors (requires external data)
4. Risk mitigation: financial stability of homeowners and homes based on historic data

Make forecast predictions for zip codes and their mean home values using Seasonal ARIMA time-series analysis.


## Model

Make forecast predictions for zip codes and their mean home values using Seasonal ARIMA time-series analysis.

**Model Identification**
* Plots and summary statistics
* Identify trends, seasonality, and autoregression elements
* Get an idea of the amount of differencing and size of lag that will be required.

**Parameter Estimation**
* Use a fitting procedure to find the coefficients of the regression model.
* split data into train and test sets.

**Model Checking**
* Plots and statistical tests of the residual errors
* Determine the amount and type of temporal structure not captured by the model.

The process is repeated until a desirable level of fit is achieved on the in-sample or out-of-sample observations (e.g. training or test datasets).

**Forecasting**
* Input complete time-series and get prediction values
* Identify top 5 zip codes based on required criteria (above).



## Workflow

IMPORT
- Import libraries, functions and dataset

PREPROCESSING
- Reshape data format (Wide to Long): `pd.melt()` function
- convert datetime column to datetime objects
- set datetime col to index

RESAMPLING
- Groupby state, county, cities (`get_groups()`)
- filter by zipcodes in selected New York cities in Westchester County
- EDA & visualizations

PARAMETERS
- seasonal decompositon
- decomposed residuals
- plot time series
- Check for stationarity
- pmdarima (differencing)

INITIAL MODEL
- Split train/test by integer index based on len(data)//test_size
- train ARIMA model
- Model results/summary for each independent zipcode

REVISED MODEL
- Seasonal Arima (SARIMA)
- Exogenous Data
- SARIMAX

FORECAST
- Get predictions for test set.
- Get another set of predictions built off of train+test set combined.

INTERPRET
- Analyze Results
- Summarize Findings
- Make Recommendations


## Mean Values by Train Line

Since commute time to Grand Central Station is part of the client's required criteria, I first had to look up which towns/zip codes were on which train lines. Grand Central has 3 main lines on Metro North Railroad: Hudson, Harlem, and New Haven. The first question I was interested in answering was if the average home prices for the zip codes that fall under these geographic sections display any trends. 

![](./meanvalues_area.png)

### New Haven line

Note that this does not include zip codes in Connecticut (which the New Haven line covers) since the client is only interested in towns in New York state. 
![](./newhaven_mapTime.png)

### Harlem Line

![](./harlem_mapTime.png)

### Hudson Line

![](./hudson_mapTime.png)

## Custom Time Series Analysis Functions (python)
To generate the plots above (as well as others) I wrote a couple of custom functions in python:

* mapTime()

#### `mapTime()`

```
def mapTime(d, xcol, ycol='MeanValue', X=None, vlines=None, MEAN=True):
    """
    'Maps' a timeseries plot of zipcodes 
    
    # fig,ax = mapTime(d=HUDSON, xcol='RegionName', ycol='MeanValue', MEAN=True, vlines=None)
    
    **ARGS
    d: takes a dictionary of dataframes OR a single dataframe
    xcol: column in dataframe containing x-axis values (ex: zipcode)
    ycol: column in dataframe containing y-axis values (ex: price)
    X: list of x values to plot on x-axis (defaults to all x in d if empty)
    
    **kw_args
    mean: plots the mean of X (default=True)
    vlines : default is None: shows MIN_, MAX_, crash 
    
    *Ex1: `d` = dataframe
    mapTime(d=NY, xcol='RegionName', ycol='MeanValue', X=list_of_zips)
    
    *Ex2: `d` = dictionary of dataframes
    mapTime(d=NYC, xcol='RegionName', y='MeanValue')
    """
  
    
    # create figure for timeseries plot
    fig, ax = plt.subplots(figsize=(21,13))
    plt.title(label=f'Time Series Plot: {str(ycol)}')
    ax.set(title='Mean Home Values', xlabel='Year', ylabel='Price($)', font_dict=font_title)  
    
    zipcodes = []
    #check if `d` is dataframe or dictionary
    if type(d) == pd.core.frame.DataFrame:
        # if X is empty, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d[xcol].unique())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        for zc in zipcodes:
            if zc < breakpoint:
                ls='-'
            else:
                ls='--'
            ts = d[zc][ycol].rename(zc)#.loc[zc]
            ts = d[ycol].loc[zc]
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls)
        ## Calculate and plot the MEAN
        
        if MEAN:
            mean = d[ycol].mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
    
    elif type(d) == dict:
        # if X passed in as empty list, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d.keys())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        # create empty dictionary for plotting 
        txd = {}
        # create different linestyles for zipcodes (easier to distinguish if list is long)
        for i,zc in enumerate(zipcodes):
            if i < breakpoint:
                ls='-'
            else:
                ls='--'
            # store each zipcode as ts  
            ts = d[zc][ycol].rename(zc)
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls, lw=2);
            txd[zc] = ts
            
        if MEAN:
            mean = pd.DataFrame(txd).mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
            
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
            
    if vlines:
        ## plot crash, min and max vlines
        crash = '01-2009'
        ax.axvline(crash, label='Housing Index Drops',color='red',ls=':',lw=2)
        MIN_ = ts.loc[crash:].idxmin()
        MAX_ = ts.loc['2004':'2010'].idxmax()
        ax.axvline(MIN_, label=f'Min Price Post Crash {MIN_}', color='black',lw=2)    
        ax.axvline(MAX_,label='Max Price', color='black', ls=':',lw=2) 

    return fig, ax
```

## Seasonality and Trends

To check for seasonality and remove trends, I also wrote a custom function to generate all the necessary time series analysis plots in one shot. This is really handy for not having to repeat steps over and over again:

![](./output_50_1.png)
![](./output_50_2.png)
![](./output_50_3.png)
![](./output_50_4.png)
![](./output_50_5.png)
![](./output_50_6.png)
![](./output_65_0.png)

All of the above are generated with the clockTime() function:

```
def clockTime(ts, lags, d, TS, y):
    """    
     /\    /\    /\    /\  ______________/\/\/\__-_-_
    / CLOCKTIME STATS /  \/
        \/    \/    \/    

    # clockTime(ts, lags=43, d=5, TS=NY, y='MeanValue',figsize=(13,11))
    #
    # ts = df.loc[df['RegionName']== zc]["MeanValue"].rename(zc).resample('MS').asfreq()
    """
    # import required libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import log 
    import pandas as pd
    from pandas import Series
    from pandas.plotting import autocorrelation_plot
    from pandas.plotting import lag_plot
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf  
    
    print(' /\\   '*3+' /')
    print('/ CLOCKTIME STATS')
    print('    \/'*3)

    #**************#   
    # Plot Time Series
    #original
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    ts.plot(label='Original', ax=axes[0,0],c='red')
    # autocorrelation 
    autocorrelation_plot(ts, ax=axes[0,1], c='magenta') 
    # 1-lag
    autocorrelation_plot(ts.diff().dropna(), ax=axes[1,0], c='green')
    lag_plot(ts, lag=1, ax=axes[1,1])
    
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    # DICKEY-FULLER Stationarity Test
    # TS = NY | y = 'MeanValue'
    dtest = adfuller(TS[y].dropna())
    if dtest[1] < 0.05:
        ## difference data before checking autoplot
        stationary = False
        r = 'rejected'
    else:
        ### skip differencing and check autoplot
        stationary = True 
        r = 'accepted'

    #**************#
    # ts orders of difference
    ts1 = ts.diff().dropna()
    ts2 = ts.diff().diff().dropna()
    ts3 = ts.diff().diff().diff().dropna()
    ts4 = ts.diff().diff().diff().diff().dropna()
    tdiff = [ts1,ts2,ts3,ts4]
    # Calculate Standard Deviation of Differenced Data
    sd = []
    for td in tdiff:
        sd.append(np.std(td))
    
    #sd = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD = pd.DataFrame(data=sd,index=['ts1',' ts2', 'ts3', 'ts4'], columns={'sd'})
    #SD['sd'] = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD['D'] = ['d=1','d=2','d=3','d=4']
    MIN = SD.loc[SD['sd'] == np.min(sd)]['sd']

    # Extract and display full test results 
    output = dict(zip(['ADF Stat','p-val','# Lags','# Obs'], dtest[:4]))
    for key, value in dtest[4].items():
        output['Crit. Val (%s)'%key] = value
    output['min std dev'] = MIN
    output['NULL HYPOTHESIS'] = r
    output['STATIONARY'] = stationary
     
    # Finding optimal value for order of differencing
    from pmdarima.arima.utils import ndiffs
    adf = ndiffs(x=ts, test='adf')
    kpss = ndiffs(x=ts, test='kpss')
    pp = ndiffs(x=ts, test='pp')
        
    output['adf,kpss,pp'] = [adf,kpss,pp]

    #**************#
    # show differencing up to `d` on single plot (default = 5)
    fig2 = plt.figure(figsize=(13,5))
    ax = fig2.gca()
    for i in range(d):
        ax = ts.diff(i).plot(label=i)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # DIFFERENCED SERIES
    fig3 = plt.figure(figsize=(13,5))
    ts1.plot(label='d=1',figsize=(13,5), c='blue',lw=1,alpha=.7)
    ts2.plot(label='d=2',figsize=(13,5), c='red',lw=1.2,alpha=.8)
    ts3.plot(label='d=3',figsize=(13,5), c='magenta',lw=1,alpha=.7)
    ts4.plot(label='d=4',figsize=(13,5), c='green',lw=1,alpha=.7)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True, 
               fancybox=True, facecolor='lightgray')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    #**************#
    
    # Plot ACF, PACF
    fig4,axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    plot_acf(ts1,ax=axes[0,0],lags=lags)
    plot_pacf(ts1, ax=axes[0,1],lags=lags)
    plot_acf(ts2,ax=axes[1,0],lags=lags)
    plot_pacf(ts2, ax=axes[1,1],lags=lags)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # plot rolling mean and std
    #Determine rolling statistics
    rolmean = ts.rolling(window=12, center=False).mean()
    rolstd = ts.rolling(window=12, center=False).std()
        
    #Plot rolling statistics
    fig = plt.figure(figsize=(13,5))
    orig = plt.plot(ts, color='red', label='original')
    mean = plt.plot(rolmean, color='cyan', label='rolling mean')
    std = plt.plot(rolstd, color='orange', label='rolling std')
    
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
    plt.title('Rolling mean and standard deviation')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # # Check Seasonality 
    """
    Calculates and plots Seasonal Decomposition for a time series
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomp = seasonal_decompose(ts, model='additive') # model='multiplicative'

    decomp.plot()
    ts_seas = decomp.seasonal

    ax = ts_seas.plot(c='green')
    fig = ax.get_figure()
    fig.set_size_inches(13,11)

    ## Get min and max idx
    min_ = ts_seas.idxmin()
    max_ = ts_seas.idxmax()
    min_2 = ts_seas.loc[max_:].idxmin()

    ax.axvline(min_,label=min_,c='red')
    ax.axvline(max_,c='red',ls=':', lw=2)
    ax.axvline(min_2,c='red', lw=2)

    period = min_2 - min_
    ax.set_title(f'Season Length = {period}')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
   
    #*******#
    clock = pd.DataFrame.from_dict(output, orient='index')
    print(' /\\   '*3+' /')
    print('/ CLOCK-TIME STATS')
    print('    \/'*3)
    
    #display results
    print('---'*9)
    return clock

```


## GridsearchCV with SARIMAX

I then ran a gridsearch using a Seasonal ARIMA (SARIMAX) model to make forecast predictions on all 66 zip codes in Westchester County.

Using PANDAS, I narrowed down the list by top 10 highest ROI zip codes. I then identified which of these had the lowest confidence intervals in order to ensure I was only selecting the most accurate results.

![](./conf_roi_pred_3D.png)

![](./conf_roi_heatmap.png)

The top five I selected based on the above criteria were 10549, 10573, 10604, 10605, 10706:

![](./10549.png)
![](./10573.png)
![](./10604.png)
![](./10605.png)
![](./10706.png)

![](./top5_final_mapTime.png)

## Contact

If you want to contact me you can reach me at <rukeine@gmail.com>.

## License

This project uses the following license: [MIT License](./LICENSE.md).
