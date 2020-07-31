#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan-Niklas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc

import scipy.stats as stats


import random
import numpy as np
from mpmath import gamma
import math
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray
import pystan
import arviz as az
from pandas import DataFrame
import seaborn as sns





data = pd.read_csv("Add path of file here", delim_whitespace=True,header=None)

# Remove the first line - customerid of the big dataset
data = data.drop(data.columns[[0]], axis=1)

# Name the columns
data.columns = ['customerid', 'date', 'numberofcds', 'dollarvalueoftransaction']

# Change format of date
data['date'] = data['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))


# Get the number of purchases each customer
number = data.groupby(['customerid']).size()

number = pd.DataFrame(number).reset_index()
number.columns = ['customerid', 'X_h']


datac = pd.merge(data, number, on='customerid')



# Get the earliest and latest date of the purchase for each customer 
datemin = data.loc[data.groupby(["customerid"]).date.idxmin()]
datemin = datemin.drop(datemin.columns[[2, 3]], axis=1)
datemin.columns = ['customerid', 'earliestdate']
datemin.reset_index(inplace = True, drop = True)


datemax = data.loc[data.groupby(["customerid"]).date.idxmax()]
datemax = datemax.drop(datemax.columns[[2, 3]], axis=1)
datemax.columns = ['customerid', 'latestdate']
datemax.reset_index(inplace = True, drop = True)

# Append t_h to datemin
th_dif = (datemax['latestdate'] - datemin['earliestdate']) / np.timedelta64(1, 'M')
th_dif = th_dif.to_frame()
th_dif.columns = ['t_h']
datemin['t_h']= th_dif['t_h']

# Get the last date of a purchase, which is 1998-06-30 and calculate T_h for every customer
lastdate = data.loc[data['date'].idxmax()].to_frame()
lastdate = lastdate.iat[1,0]

datemin['T_h'] = (lastdate - datemin['earliestdate']) / np.timedelta64(1, 'M')

# Complete dataset with X_h,t_h,T_h
datac = pd.merge(datac,datemin[['customerid','t_h']],on='customerid', how='left')
datac = pd.merge(datac,datemin[['customerid','T_h']],on='customerid', how='left')

datac['NumberPurchasesPerUnit']= (datac['X_h'])/(datac['T_h'])
#### NumberPurchasesPerUnit all values between 0 and 1 are set to 1
datac.loc[(datac['NumberPurchasesPerUnit'] >= 0) & (datac['NumberPurchasesPerUnit'] <= 1), 'NumberPurchasesPerUnit'] = 1 
#### Round the values of NumberPurchasesPerUnit
datac['NumberPurchasesPerUnitRounded']= round(datac["NumberPurchasesPerUnit"])
datac['NumberPurchasesPerUnitRounded']= datac['NumberPurchasesPerUnitRounded'].astype(np.int64)

# X_h for each customer data.iloc[0,n]
datad = datac.drop_duplicates('customerid')
datad.reset_index(inplace = True, drop = True)
X_h = datad.drop(datad.columns[[0,1,2,3,5,6]], axis=1)
t_h = datad.drop(datad.columns[[0,1,2,3,4,6]], axis=1)
T_h = datad.drop(datad.columns[[0,1,2,3,4,5]], axis=1)


#  Contains the TotalSum for every customer

TotalSum = datac.groupby('customerid')['dollarvalueoftransaction'].agg('sum')



Amount = datac
Amount['Time'] = pd.to_datetime(Amount['date'])
Amount['Rank'] = Amount.groupby('customerid')['Time'].rank(ascending=True)


Amount=Amount.sort_values(['customerid', 'Rank'], ascending=[True, True])
Amount = Amount.drop(Amount.columns[[1,2,4,5,6,7,8,9]], axis=1)
Amount= Amount.rename(columns={'dollarvalueoftransaction': 'Amount'})
Amount = Amount.astype({"Rank": int})
Amount = Amount[['customerid', 'Rank', 'Amount']]


# Calculate interpurchase times for all customers
datacIPT = datac
datacIPT['IPT_days'] = datacIPT.sort_values(['customerid','Time']).groupby('customerid')['Time'].diff()

# Drop rows with Rank 1 and NA/ occasions on the same day
datacIPT = datacIPT[datacIPT.Rank != 1]
datacIPT = datacIPT.dropna(subset=['IPT_days'])

na = datacIPT[datacIPT.isna().any(axis=1)]

# Calculate interpurchase time in months
datacIPT["IPT_months"] = datacIPT["IPT_days"].dt.days/31





# Calculate Mean, Standard Deviation, Minimum and maximum of IPT, Amount, X_h, t_h and T_h

# IPT 

IPT_mean = datacIPT["IPT_months"].mean()
IPT_std = datacIPT["IPT_months"].std()
IPT_min = datacIPT["IPT_months"].min()
IPT_max = datacIPT["IPT_months"].max()

# Amount

Amount_mean = datac["dollarvalueoftransaction"].mean()
Amount_std = datac["dollarvalueoftransaction"].std()
Amount_min = datac["dollarvalueoftransaction"].min()
Amount_max = datac["dollarvalueoftransaction"].max()

# X_h

X_h_mean = X_h["X_h"].mean()
X_h_std = X_h["X_h"].std()
X_h_min = X_h["X_h"].min()
X_h_max = X_h["X_h"].max()

# t_h

t_h_mean = t_h["t_h"].mean()
t_h_std = t_h["t_h"].std()
t_h_min = t_h["t_h"].min()
t_h_max = t_h["t_h"].max()


# T_h

T_h_mean = T_h["T_h"].mean()
T_h_std = T_h["T_h"].std()
T_h_min = T_h["T_h"].min()
T_h_max = T_h["T_h"].max()











################## Plots

count = datac.groupby('date').date.count()
count.columns = ['date', 'NumberOfPurchases']

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

#Plot Number of Purchases per day
# NumberPurchases = count.plot.line(title="Number of purchases per day")
# NumberPurchases.set_xlabel("Date")
# NumberPurchases.set_ylabel("Number of purchases")
# NumberPurchases.set_facecolor('lightgrey')


# Plot Interpurchase time frequency 


# Plot_IPT = datacIPT['IPT_days'].astype('timedelta64[D]').plot.hist(title="Interpurchase time in days (with bins=100)",bins=100)
# Plot_IPT.set_xlabel("IPT in days")
# Plot_IPT.set_facecolor('lightgrey')




# Plot number of CDS and Purchases Amount




# AmountPlot = datac["dollarvalueoftransaction"].plot.hist(title="Purchase amount per transaction",bins=100)
# AmountPlot.set_xlabel("Purchase amount per transaction in Dollar")
# #AmountPlot.set_ylabel("Number of purchases")
# AmountPlot.set_facecolor('lightgrey')


# Plot Purchase Amount per customer from first to last observed purchase

# AmountPlotCustomer = TotalSum.plot.hist(title="Purchase amount per customer",bins=100)
# AmountPlotCustomer.set_xlabel("Purchase amount in Dollar per customer from first to last observed purchase")
# #AmountPlotCustomer.set_ylabel("Number of purchases")
# AmountPlotCustomer.set_facecolor('lightgrey')




# Plot Purchases Per Unit 


# PurchasesPerUnit = datad["NumberPurchasesPerUnitRounded"].plot.hist(title="Purchases Per Time unit",bins=100)
# PurchasesPerUnit.set_xlabel("Purchases Per Unit")
# # #AmountPlot.set_ylabel("Number of purchases")
# PurchasesPerUnit.set_facecolor('lightgrey')


# Boxplot t_h and T_h


# Boxplot = datac.boxplot(column=['t_h','T_h'], return_type='axes');

# Boxplot.set_ylabel('Time in Months')
# Boxplot.set_title('Comparison of t_h and T_h')


#Boxplot X_h

# Boxplot = datad.boxplot(column=['X_h'], return_type='axes');
# Boxplot.set_ylabel('Number of purchase occasions')
# Boxplot.set_title('Boxplot for X_h')



# Boxplot for numberofcds 

# Boxplot = datac.boxplot(column=['numberofcds'], return_type='axes');

# Boxplot.set_title('Number of CDs')


# Boxplot for Purchase Amount

# Boxplot = datac.boxplot(column=['dollarvalueoftransaction'], return_type='axes');

# Boxplot.set_title('Purchase Amount')
# Boxplot.set_ylabel('Amount in $')









# Re = pd.DataFrame()
# Re["t_h"] = t_h["t_h"].values
# Re["totlife_mean"] = Re["t_h"]+5
# Re["lambda_mean"] = Re["t_h"]
# Re["theta_mean"] = Re["t_h"]
# Re["sigmaw_mean"] = Re["t_h"]-5

# # Remaining contains the remaining lifetime of a customer

# Re['remaining'] = Re['totlife_mean'].sub(Re['t_h'], axis = 0)




# # Remaining contains the remaining lifetime of a customer, if its bigger than T_h
# Re["T_h"] = T_h["T_h"].values
# Re['remaining_T_h'] = Re['totlife_mean'].sub(Re['T_h'], axis = 0)

# # Show rows with remaining bigger than 0
# T= Re.loc[Re['remaining_T_h'] >= 0]
