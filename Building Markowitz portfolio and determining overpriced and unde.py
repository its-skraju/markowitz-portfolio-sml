#!/usr/bin/env python
# coding: utf-8

# <h1><center>IME 611A Financial Engineering, IIT Kanpur</center></h1>
# <p><h1><center>Course Project A</center></h1></p>
# <p><center>Akash Sonowal (20114003); Sk Raju (20114014)</center></p>
# <p><center>Group 15 (MTech IME, 1st year)</center></p>
# 
# 

# In[1]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime


# Loading Dataset

# In[2]:


data = pd.read_csv('C:/Users/MSI-PC/Documents/DataSets/projectA_data.csv')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.tail()


# Making Date as index

# In[6]:


data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.index


# In[7]:


data.head()


# Checking for missing values

# In[8]:


features_with_na=[features for features in data.columns if data[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(data[feature].isnull().mean()*100, 4),  ' % missing values')


# In[9]:


data = data.dropna(axis=0,subset=['^NSEI'])


# In[10]:


data = data.dropna(axis=1, how='any')


# In[11]:


data.shape


# In[12]:


features_with_na=[features for features in data.columns if data[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(data[feature].isnull().mean()*100, 4),  ' % missing values')


# In[13]:


data.head()


# In[ ]:





# ### Task 1: Plot the daily stock price of each of these companies. Present the time-series characteristics of these prices. Construct total return and log total return series. Plot the return series and present their time-series characteristics.

# In[14]:


df = data.copy()


# In[15]:


import statistics
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


#Plot of daily stock price and time-series characteristics
fig, axes = plt.subplots(nrows=len(df.columns), ncols=1,figsize=(15,500))
for i in range(len(df.columns)):
    axes[i].plot(df[df.columns[i]], label='Observed Stock Price')
    cycle, trend = sm.tsa.filters.hpfilter(df[df.columns[i]]) #returns cycle and trend
    axes[i].plot(cycle, label="cycle")
    axes[i].plot(trend, label="trend")
    axes[i].set_xlabel('time')
    axes[i].set_ylabel('stock close price')
    axes[i].set_title(df.columns[i])
    axes[i].legend()
fig.tight_layout(pad=3.0)


# In[17]:


df_seasonal = df.copy()


# In[18]:


df_seasonal['month'] = df_seasonal.index.month


# In[19]:


#Monthwise seasonality
for i in df_seasonal.columns.difference(['month']):
    plt.figure(figsize=(18,5))
    df_seasonal.groupby(['month'])[i].mean().plot.bar()
    plt.xlabel('month')
    plt.ylabel("average closing price monthwise")
    plt.title("Stock history")
    plt.legend(loc='upper center')
    plt.grid(True)


# In[20]:


#Total Return
total_return = df.pct_change(1)
total_return.dropna(axis=0, inplace=True)


# In[21]:


#Plotting total return
fig, axes = plt.subplots(nrows=len(total_return.columns), ncols=1,figsize=(15,500))
for i in range(len(total_return.columns)):
    axes[i].plot(total_return[total_return.columns[i]], label='Observed total return')
    cycle, trend = sm.tsa.filters.hpfilter(total_return[total_return.columns[i]]) #returns cycle and trend
    axes[i].plot(cycle, label="cycle")
    axes[i].plot(trend, label="trend")
    axes[i].set_xlabel('time')
    axes[i].set_ylabel('stock total return')
    axes[i].set_title(total_return.columns[i])
    axes[i].legend()
fig.tight_layout(pad=3.0)


# In[22]:


total_return_seasonal = total_return.copy()


# In[23]:


total_return_seasonal['month'] = total_return_seasonal.index.month


# In[24]:


#Monthwise seasonality of total return
for i in total_return_seasonal.columns.difference(['month']):
    plt.figure(figsize=(18,5))
    total_return_seasonal.groupby(['month'])[i].mean().plot.bar()
    plt.xlabel('month')
    plt.ylabel("average total return monthwise")
    plt.title("Stock history")
    plt.legend(loc='upper center')
    plt.grid(True)


# In[25]:


#Log total return
log_return = np.log(df/df.shift(1))
log_return.dropna(axis=0, inplace=True)


# In[26]:


#Plotting log return
fig, axes = plt.subplots(nrows=len(log_return.columns), ncols=1,figsize=(15,500))
for i in range(len(log_return.columns)):
    axes[i].plot(log_return[log_return.columns[i]], label='Observed log return')
    cycle, trend = sm.tsa.filters.hpfilter(log_return[log_return.columns[i]]) #returns cycle and trend
    axes[i].plot(cycle, label="cycle")
    axes[i].plot(trend, label="trend")
    axes[i].set_xlabel('time')
    axes[i].set_ylabel('stock log return')
    axes[i].set_title(log_return.columns[i])
    axes[i].legend()
fig.tight_layout(pad=3.0)


# In[27]:


log_return_seasonal = log_return.copy()


# In[28]:


log_return_seasonal['month'] = log_return_seasonal.index.month


# In[29]:


#Monthwise seasonality of log return
for i in log_return_seasonal.columns.difference(['month']):
    plt.figure(figsize=(18,5))
    log_return_seasonal.groupby(['month'])[i].mean().plot.bar()
    plt.xlabel('month')
    plt.ylabel("average log return monthwise")
    plt.title("Stock history")
    plt.legend(loc='upper center')
    plt.grid(True)


# In[ ]:





# ### Task 2: Check the distribution of these returns. Present test statistics whether the returns are following a normal distribution. If not, then fit other distribution types which fit the data better

# In[30]:


from scipy.stats import norm


# In[31]:


for feature in total_return:
    fig= plt.figure(figsize=(15,5))
    #plotting histogram
    total_return[feature].hist(bins=100, color='deepskyblue', label= feature)
    
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title("Total Return histogram")
    plt.legend(loc='upper right')
    
    #plotting normal distribution
    mean, std = norm.fit(total_return[feature])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label = "Fit results: mean = %.2f,  std = %.2f" % (mean, std))
    
    plt.show()


# In[32]:


for feature in log_return:
    fig= plt.figure(figsize=(15,5))
    #plotting histogram
    log_return[feature].hist(bins=100, color='deepskyblue', label= feature)
    
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title("Log Return histogram")
    plt.legend(loc='upper right')
    
    #plotting normal distribution
    mean, std = norm.fit(log_return[feature])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label = "Fit results: mean = %.2f,  std = %.2f" % (mean, std))
    
    plt.show()


# In[33]:


#Test statistics for normal distribution for total returns
for i in total_return.columns:
    k2, p = stats.normaltest(total_return[i])
    alpha = 0.05
    print(k2, p)
    if p < alpha: # null hypothesis: Our stocks comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


# In[34]:


#Test statistics for normal distribution for log returns
for i in log_return.columns:
    k2, p = stats.normaltest(log_return[i])
    alpha = 0.05
    print(k2, p)
    if p < alpha: # null hypothesis: Our stocks comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


# In[35]:


#For fitting distributions we have picked only 4 stocks of different volatility
dist_list = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2',
             'cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife',
             'fisk','foldcauchy','foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme',
             'gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l',
             'halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull',
             'johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace',
             'lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3',
             'powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss',
             'semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises',
             'vonmises_line','wald','weibull_min','weibull_max']
len(dist_list)


# In[36]:


#Fitting total returns
results = []
stocks_fit_dist = ['TATAMOTORS.NS','SBIN.NS','SIEMENS.NS','HINDUNILVR.NS','^NSEI']
for i in stocks_fit_dist:
    print("**********************************************************************************")
    print("The distributions test for", i)
    print("**********************************************************************************")
    for j in dist_list:
        dist = getattr(stats, j)
        param = dist.fit(total_return[i])
        a = stats.kstest(total_return[i], j, args=param)
        results.append((j,a[0],a[1]))    
    results.sort(key=lambda x:float(x[2]), reverse=True)
    for k in results:
        print("{}: statistic={}, pvalue={}".format(k[0], k[1], k[2]))


# In[37]:


#Fitting log total returns
results = []
stocks_fit_dist = ['TATAMOTORS.NS','SBIN.NS','SIEMENS.NS','HINDUNILVR.NS','^NSEI']
for i in stocks_fit_dist:
    print("**********************************************************************************")
    print("The distributions test for", i)
    print("**********************************************************************************")
    for j in dist_list:
        dist = getattr(stats, j)
        param = dist.fit(log_return[i])
        a = stats.kstest(log_return[i], j, args=param)
        results.append((j,a[0],a[1]))    
    results.sort(key=lambda x:float(x[2]), reverse=True)
    for k in results:
        print("{}: statistic={}, pvalue={}".format(k[0], k[1], k[2]))


# In[38]:


import math
import scipy.stats as scs
def main(r):
    a, b, loc, scale = scs.johnsonsu.fit(r) # fit the data and get distribution parameters back
    #γ would be a, δ would be b; ξ would be loc, λ would be scale
    return a, b  


# In[39]:


#Fitting distribution to total return
from scipy.stats import johnsonsu
#Total Returns
stocks_fit_dist = ['TATAMOTORS.NS','SBIN.NS','SIEMENS.NS','HINDUNILVR.NS','^NSEI']
for i in stocks_fit_dist:
    fig= plt.figure(figsize=(15,5))

    a, b = main(total_return[i])
    x = np.linspace(johnsonsu.ppf(0.01, a, b),johnsonsu.ppf(0.99, a, b), 100)
    plt.plot(x, johnsonsu.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='johnsonsu pdf')
    plt.xlabel('Total returns')
    plt.ylabel('Frequencies')
    plt.title(i)
    plt.legend(loc='upper right')
    
    #total_return[i].hist(bins=100, color='deepskyblue', label= i)
    plt.show()


# In[40]:


#Fitting distribution to log total return
from scipy.stats import johnsonsu
#Total Returns
stocks_fit_dist = ['TATAMOTORS.NS','SBIN.NS','SIEMENS.NS','HINDUNILVR.NS','^NSEI']
for i in stocks_fit_dist:
    fig= plt.figure(figsize=(15,5))

    a, b = main(log_return[i])
    x = np.linspace(johnsonsu.ppf(0.01, a, b),johnsonsu.ppf(0.99, a, b), 100)
    plt.plot(x, johnsonsu.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='johnsonsu pdf')
    plt.xlabel('Log total returns')
    plt.ylabel('Frequencies')
    plt.title(i)
    plt.legend(loc='upper right')
    
    #log_return[i].hist(bins=100, color='deepskyblue', label= feature)
    plt.show()


# ### Task 3: Assuming the data to be following a normal distribution, estimate the returns, variances, standard deviation, variance-covariance matrix

# In[41]:


#If the data is assumed to be normal, then the annual return is calculated in the following step
annual_total_return_norm = total_return.describe().iloc[1:3,:]
annual_total_return_norm


# In[42]:


annual_total_return_norm.iloc[0,:] = (annual_total_return_norm.iloc[0,:])*252
annual_total_return_norm.iloc[1,:] = (annual_total_return_norm.iloc[1,:])*(252**0.5)
annual_total_return_norm.loc['var'] = ((annual_total_return_norm.iloc[1,:])**2)
annual_total_return_norm.loc['68% confidence_return_lower_range'] = annual_total_return_norm.loc['mean'] - 1*annual_total_return_norm.loc['std']
annual_total_return_norm.loc['68% confidence_return_upper_range'] = annual_total_return_norm.loc['mean'] + 1*annual_total_return_norm.loc['std']

annual_total_return_norm.loc['95% confidence_return_lower_range'] = annual_total_return_norm.loc['mean'] - 2*annual_total_return_norm.loc['std']
annual_total_return_norm.loc['95% confidence_return_upper_range'] = annual_total_return_norm.loc['mean'] + 2*annual_total_return_norm.loc['std']


# In[43]:


annual_total_return_norm.head(7)


# In[44]:


#Variance-Covariance matrix
total_return.cov()


# In[45]:


#Correlation between pair of stocks total return
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.figure(figsize = (36,36))
sns.heatmap(total_return.corr(), vmin=-1, vmax=1,center=0, cmap = "BrBG")
plt.show()


# ### Task 4: Construct a Markowitz portfolio and plot the efficient frontier. Prove that the portfolio frontier will have a parabolic structure.

# Select stocts for portfolio

# In[46]:


sharpe_ratio = ((log_return.mean()/log_return.std()))*(252**0.5)


# In[47]:


sharpe_ratio = pd.DataFrame(sharpe_ratio)
sharpe_ratio.head()


# In[48]:


sharpe_ratio_sorted = sharpe_ratio.sort_values(by=0, ascending=False)
sharpe_ratio_sorted.head(15)


# In[49]:


#Stocks selected based on maximum sharp ratio
selected =  ['ADANITRANS.NS','ADANIENT.NS','BAJFINANCE.NS', 'IGL.NS', 'RELIANCE.NS','MUTHOOTFIN.NS', 'NAUKRI.NS', 'HDFCBANK.NS', 'NESTLEIND.NS', 'BAJAJFINSV.NS','TITAN.NS', 'PIDILITIND.NS', 'BERGEPAINT.NS','TATACONSUM.NS','BIOCON.NS']


# In[50]:


df = log_return[selected]


# In[51]:


#Portfolio return
def portfolio_return(weights):
    return np.dot(df.mean(), weights)*252


# In[52]:


#Portfolio variance
def portfolio_variance(weights):
    return np.dot(np.dot(df.cov(), weights),weights)*252


# In[53]:


#Portfolio std
def portfolio_std(weights):
    return (np.dot(np.dot(df.cov(), weights),weights)*252)**(0.5)


# In[54]:


#Weight distribution
def weights_distribution(df):
    r = np.random.random(len(df.columns))
    r/=r.sum()
    return r


# In[82]:


returns = []
stds = []
w = []
sr = []

for i in range(7000):
    weights = weights_distribution(df)
    returns.append(portfolio_return(weights))
    stds.append(portfolio_std(weights))
    w.append(weights)
    sr.append(returns[i]/stds[i])


# In[56]:


#Draw a parabola
def parabola(x, a, alpha, beta):
    return ((4*a*(x-alpha))**0.5) + beta


# In[89]:


#Plotting Efficient frontier
plt.scatter(stds, returns,c=sr, cmap='viridis')
plt.scatter(min(stds),returns[stds.index(min(stds))], c="orange" )
plt.title("Efficient frontier")
plt.xlabel("Portfolio_std")
plt.ylabel("Portfolio_return")
plt.colorbar(label='Sharpe Ratio')

x = np.linspace(0, max(stds), 1000)
y = parabola(x,0.079, min(stds), returns[stds.index(min(stds))])
plt.plot(x, y, label='Parabola', c= 'r')  
plt.legend(loc='lower right') 

plt.show()


# In[58]:


minstd_weights = w[stds.index(min(stds))]


# ### Task 5: Use a risk-free rate of 5% and plot the Security Market Line (SML). Demonstrate a few underpriced and overpriced securities. 

# In[59]:


log_return['mean_row'] = log_return.mean(axis = 1)


# In[60]:


def beta_stocks_1(lr):
    return lr.cov().iloc[0,1]/(lr['mean_row'].var())


# In[61]:


beta_stock = []

for feature in log_return:
    lr = log_return[[feature,'mean_row']]
    beta_stock.append(beta_stocks_1(lr))


# In[62]:


def SML_1(rf,rm,label):
    betas = [x/10 for x in range(21)]
    assetReturns = [rf+(rm-rf)*x for x in betas]
    plt.plot(betas,assetReturns,label=label)
    plt.xlabel("Asset Beta")
    plt.ylabel("Asset Return")
    plt.plot(1,rm,"ro")


# In[63]:


for i in range(len(log_return.columns)):
    SML_1(.05,log_return['mean_row'].mean()*252,log_return[log_return.columns[i]])
    plt.plot(beta_stock[1],log_return[log_return.columns[i]].mean()*252,"ro", color='g')
    plt.title(log_return.columns[i])
    plt.show()


# In[64]:


log_return = log_return.drop('mean_row', axis =1)


# ### Task 6: Read about at least one other approach to estimate variance-covariance matrix and implement the same.

# We are taking Nifty50 values as market return

# In[65]:


def beta_stocks_2(lr):
    return lr.cov().iloc[0,1]/(lr['^NSEI'].var())


# In[66]:


beta_stock = []

for feature in log_return:
    lr = log_return[[feature,'^NSEI']]
    beta_stock.append(beta_stocks_2(lr))


# In[67]:


def SML_2(rf,rm,label):
    betas = [x/10 for x in range(21)]
    assetReturns = [rf+(rm-rf)*x for x in betas]
    plt.plot(betas,assetReturns,label=label)
    plt.xlabel("Asset Beta")
    plt.ylabel("Asset Return")
    plt.plot(1,rm,"ro")


# In[68]:


for i in range(len(log_return.columns)):
    SML_2(0.05,log_return['^NSEI'].mean()*252,log_return[log_return.columns[i]])
    plt.plot(beta_stock[1],log_return[log_return.columns[i]].mean()*252,"ro", color='g')
    plt.title(log_return.columns[i])
    plt.show()


# In[ ]:





# ### Task 7: For the actual return distribution that you have observed, identify the suitable return, and risk measure and highlight how those should be considered in the portfolio weight identification

# In[69]:


equal_weights = [1/15]*15


# In[70]:


print("Portfolio return with equal distribution of weights: ",portfolio_return(equal_weights))


# In[71]:


print("Portfolio risk with equal distribution of weights: ",portfolio_std(equal_weights))


# In[73]:


print("Portfolio weights with Markowitz distribution of weights: ",minstd_weights)


# In[74]:


print("Portfolio return with Markowitz distribution of weights: ",portfolio_return(minstd_weights))


# In[75]:


print("Portfolio risk with Markowitz distribution of weights: ",portfolio_std(minstd_weights))


# In[76]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# In[77]:


suitable_return = 0.36


# In[79]:


print("Risk Measure for suitable return: ",stds[returns.index(find_nearest(returns, suitable_return))])
print("Sharpe ratio for suitable return: ",sr[returns.index(find_nearest(returns, suitable_return))])
print("Weight identification for suitable return:",w[returns.index(find_nearest(returns, suitable_return))])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




