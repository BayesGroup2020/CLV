#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan-Niklas
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sc

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

## Read in data

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

ExpectedTheta =round(datac["dollarvalueoftransaction"].mean());

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






np.random.seed(1234)


## Model with custom lomax function for total lifetime of a customer

clv_model = """functions {
  // Define log probability density function
  real customLomax_lpdf(real totLife_h, real s, real beta) {
    return log(s)-log(beta)+(s+1)*(log(beta)-log(beta+totLife_h));
  }
  real customProbability_lpmf(int x,real r,real alpha, real T_h){
  return log(tgamma(r+x))-log(tgamma(r))-log(tgamma(x+1))+r*(log(alpha)-log(alpha+T_h))+x*(log(T_h)-log(alpha+T_h));
  }
  }
  data {
  int<lower=0> N;  // number of customers
  real<lower=0> t_h[N];  // t_h;
  real<lower=0> T_h[N];  // T_h;
  int<lower=0> X_h[N];  // X_h;
  int <lower=0> NumberPurchases[N]; // number of purchasesPerUnitTime_h
  real <lower=0> TotalSum[N]; //
  
  //Segments
  int<lower=0> O;   // # observations
  vector[O] Amount;      // Amount

}
  
parameters {
  vector <lower=0> [N]  totalife;  // totallifetime
  real <lower=0> s;
  real <lower=0> beta;
  real <lower=0> alpha;
  real <lower=0> r;
  
  real <lower=0> sigma_w_squared[N]; // Within customer variance
  real <lower=0> sigma_a_squared; // Across customer variance
  real ExpectedTheta; // Average value of order
  vector <lower=0> [N]  theta_h;  // Theta_h
  vector <lower=0> [N]  lambda_h;  // lambda_h
  
  
  
}
transformed parameters {
  vector[N] totlife;
  
  for (n in 1:N) {
    totlife [n] = t_h[n] + totalife[n];
    
}
  
}

model {
// Amount Model

int pos;
pos = 1;

//hyperpriors sigma_w_squared,sigma_a_squared,ExpectedTheta 

sigma_w_squared ~ inv_gamma(2.5, 0.5);
sigma_a_squared ~ inv_gamma(2.5, 0.5);
ExpectedTheta ~ normal(0, 100);


    for (n in 1:N) {
     segment(Amount, pos, X_h[n]) ~ normal(theta_h[n], sigma_w_squared[N]);
     pos = pos + X_h[n];
     theta_h[n] ~ normal(ExpectedTheta,sigma_a_squared);
    }

       
//hyperpriors s,beta,r,alpha
       
  s ~ gamma(5, 5);
  beta ~ gamma(5, 5);
  r ~ gamma(5,5);
  alpha ~ gamma(5,5);
  
for (n in 1:N){
 lambda_h[n] ~ gamma(r,alpha);
 totlife[n] ~ customLomax_lpdf(s, beta);
 X_h[n] ~ customProbability_lpmf(r,alpha, T_h[n]);
}
  





// Likelihood function
  
for (n in 1:N){
  
    if(totlife[n]<T_h[n])
        target += log(tgamma(r+X_h[n]))+r*log(alpha)-log(tgamma(r))-(r+X_h[n])*log(tgamma(alpha + totlife[n]));
    else
        target += log(tgamma(r+X_h[n]))+r*log(alpha)-log(tgamma(r))-(r+X_h[n])*log(tgamma(alpha + T_h[n]));

}
}

generated quantities{
    

//Count the number of interpurchase times for every customer
real counter;
real iteration;

//Purchase Amount of the future purchases for every customer 

vector[N] amount;

real remaining [N];
real remaining_T_h [N];
real NumberOfAllPurchases[N];

real simulatedIPT;
real simulatedAmount;
vector[N] lambda_h_amount;
vector[N] theta_h_amount;
vector[N] future_amount;


real FutureTimePoint;
real discounting;




for (n in 1:N){
    
    
    remaining[n]= totlife[n]-t_h[n];
    remaining_T_h[n] = totlife[n]-T_h[n];
    
    if (remaining_T_h[n]<=0){
       
        amount[n]=TotalSum[n];
        NumberOfAllPurchases[N]=X_h[n];
        future_amount[n]=0;
        
        }
    else{
        
        FutureTimePoint = t_h[n];
        
        NumberOfAllPurchases[n]=X_h[n];
        
        future_amount[n]=0;
        
        iteration=0;
        
        counter=0;
        
        amount[n]=TotalSum[n];
     
        lambda_h_amount[n] = gamma_rng(r+X_h[n],alpha+T_h[n]);
        
        while (remaining[n]>=0 && iteration<=100){
         
                iteration = iteration +1;
                simulatedIPT = exponential_rng(lambda_h_amount[n]);
                
                
                if (counter!=0 && remaining[n]-simulatedIPT>=0 ) {
                 
                        
                        FutureTimePoint = FutureTimePoint + simulatedIPT;
                        discounting = (FutureTimePoint -T_h[n])/12;
                        discounting = 1/(1+0.05)^(discounting);
                        
                        remaining[n] = remaining[n]-simulatedIPT;
                        
                        
                        
                        theta_h_amount[n] = normal_rng((sigma_a_squared*amount[n]+sigma_w_squared[n]*ExpectedTheta)/(NumberOfAllPurchases[n]*sigma_a_squared+sigma_w_squared[n]),sigma_a_squared*sigma_w_squared[n]/(NumberOfAllPurchases[n]*sigma_a_squared+sigma_w_squared[n]));
                        

                        simulatedAmount = normal_rng(theta_h_amount[n],sigma_w_squared[n]);
                        
                        simulatedAmount = simulatedAmount*discounting;
                        
                        amount[n]=amount[n]+simulatedAmount;
                        
                        future_amount[n]=future_amount[n]+simulatedAmount;
                
                        counter=counter+1;
                        
                        NumberOfAllPurchases[n]=NumberOfAllPurchases[n]+1;
                
                
                }
                
        
                if (counter==0 && t_h[n]+simulatedIPT>=T_h[n] && remaining[n]-simulatedIPT>=0 ) {
                        
                        FutureTimePoint = t_h[n] + simulatedIPT;
                        discounting = (FutureTimePoint -T_h[n])/12;
                        discounting = 1/(1+0.05)^(discounting);
                        
                
                
                        remaining[n] = remaining[n]-simulatedIPT;
        
        
                        theta_h_amount[n] = normal_rng((sigma_a_squared*amount[n]+sigma_w_squared[n]*ExpectedTheta)/(NumberOfAllPurchases[n]*sigma_a_squared+sigma_w_squared[n]),sigma_a_squared*sigma_w_squared[n]/(NumberOfAllPurchases[n]*sigma_a_squared+sigma_w_squared[n]));
                        

                        simulatedAmount = normal_rng(theta_h_amount[n],sigma_w_squared[n]);
        
                        simulatedAmount = simulatedAmount*discounting;
        
                        amount[n]=amount[n]+simulatedAmount;
                        
                        future_amount[n]=future_amount[n]+simulatedAmount;
                
                        counter=counter+1;
                        
                        NumberOfAllPurchases[n]=NumberOfAllPurchases[n]+1;
                        
                        
                
                
                }
             
        
    
                
        }
     
        
        
}

    
  }   
}


"""










## Specify data
clv_data = {'O': 6919,
            'N': 2357,
            "ExpectedTheta":ExpectedTheta,
               'y': datac['customerid'].values,
               't_h': t_h['t_h'].values,
               'T_h': T_h['T_h'].values,
               'X_h': X_h['X_h'].values,
               'Amount': Amount["Amount"].values,
               'NumberPurchases': datad['NumberPurchasesPerUnitRounded'].values,
               'TotalSum': TotalSum.values}






sm = pystan.StanModel(model_code=clv_model)
fit = sm.sampling(data=clv_data, iter=1000, chains=4)



## Diagnostics

az.plot_density(fit, var_names=["s", "beta","r","alpha","totlife"]);
az.plot_trace(fit, var_names=["s", "beta","r","alpha"]);

az.summary(fit, var_names=["s", "beta","r","alpha"]);


# Plot trace of future_amount


# FutureAmount_Summary = az.summary(fit, var_names=["amount"])
# FutureAmount_mean = FutureAmount_Summary["mean"]
# Re["FutureAmount_mean"] = FutureAmount_mean.values





# FutureAmount_Summary = az.summary(fit, var_names=["remaining_T_h"])
# FutureAmount_mean = FutureAmount_Summary["mean"]
# Re["FutureAmount_mean"] = FutureAmount_mean.values



# FutureAmount_Summary = az.summary(fit, var_names=["future_amount"])
# FutureAmount_mean = FutureAmount_Summary["mean"]
# Re["FutureAmount_mean"] = FutureAmount_mean.values


# # Show customers with a positive future purchase amount

# Re[Re.FutureAmount_mean>0]


# simulatedAmount = normal_rng(theta_h[n],sigma_w_squared[n]);

# Plot trace of future amount for specific customers

 #az.plot_trace(fit, var_names=["future_amount"], coords={'future_amount_dim_0': [5,34, 85,115,132]});


### Further plots
## Grafik 2 100 
#az.plot_trace(fit, var_names=["ExpectedTheta","sigma_a_squared"]);
## Grafik 3 100
#az.plot_density(fit, var_names=["ExpectedTheta","sigma_a_squared"]);











######## Test and Analyse


############################## Amount part

# Calculating the difference between totlife and t_h


totlife_Summary = az.summary(fit, var_names=["totlife"])
r_Summary = az.summary(fit, var_names=["r"])
alpha_Summary = az.summary(fit, var_names=["alpha"])                                     
s_Summary = az.summary(fit, var_names=["s"])                                      
beta_Summary = az.summary(fit, var_names=["beta"])   

lambda_Summary = az.summary(fit, var_names=["lambda_h"])
theta_Summary = az.summary(fit, var_names=["theta_h"])                                  
sigmaw_Summary = az.summary(fit, var_names=["sigma_w_squared"])                                     

totlife_mean = totlife_Summary["mean"]
lambda_mean = lambda_Summary["mean"]
theta_mean = theta_Summary["mean"]                                 
sigmaw_mean = sigmaw_Summary["mean"]  



Re = pd.DataFrame()
Re["t_h"] = t_h["t_h"].values
Re["totlife_mean"] = totlife_mean.values
Re["lambda_mean"] = lambda_mean.values
Re["theta_mean"] = theta_mean.values
Re["sigmaw_mean"] = sigmaw_mean.values




##############################

## Calculating the CLV


FutureAmount_Summary = az.summary(fit, var_names=["future_amount"])
FutureAmount_mean = FutureAmount_Summary["mean"]
Re["FutureAmount_mean"] = FutureAmount_mean.values


Re[Re.FutureAmount_mean>0]



Re["CLV"] = Re["FutureAmount_mean"]+TotalSum.values

###############################






# Remaining contains the remaining lifetime of a customer

Re['remaining'] = Re['totlife_mean'].sub(Re['t_h'], axis = 0)




# Remaining contains the remaining lifetime of a customer, if its bigger than T_h
Re["T_h"] = T_h["T_h"].values
Re['remaining_T_h'] = Re['totlife_mean'].sub(Re['T_h'], axis = 0)

# Show rows with remaining bigger than 0
T= Re.loc[Re['remaining_T_h'] >= 0]
T= T.reset_index()
T.rename({'index': 'customerid'}, axis=1, inplace=True)


## For the values in T we have to calculate interpurchase times and amounts
## for future purchases

# Count the number of interpurchase times for every customer
count = []

# Purchase Amount of the future purchases for every customer 

amount = []

for index, row in T.iterrows():
    
    
    remaining = T.loc[index,"remaining"]
    counter=0
    Amount=0
    
    iteration=0
    while (remaining > 0) & (iteration <= 100):
        
      IPT = np.random.exponential(scale=T["lambda_mean"],size=1)[0]
      iteration=iteration+1
      if (counter != 0) & (IPT <= remaining):
          remaining=remaining-IPT
          counter=counter+1
          Amount = Amount + np.random.normal(loc=theta_mean, scale=sigmaw_mean,size=1)[0]
      
      
      if (counter==0) & ((IPT+remaining)>= T.loc[index,"remaining_T_h"]) & (IPT <= remaining):
            remaining=remaining-IPT
            counter=counter+1
            Amount = Amount + np.random.normal(loc=theta_mean, scale=sigmaw_mean,size=1)[0]
            
    count.append(counter)
    amount.append(Amount)



T.insert(loc=0, column='NumberOfIPT', value=count)
T.insert(loc=0, column='FutureAmount', value=amount)



## Total amount per customer
calculate = pd.DataFrame(datac.groupby(['customerid'])['dollarvalueoftransaction'].agg('sum'))
calculate= calculate.reset_index()
calculate.columns = ["customerid",'Past Value']
calculate['FutureAmount']=0

## In this dataframe there are all values for the remaining customers
merged = T.merge(calculate, left_index=True, right_index=True, how='inner')

merged["total"] = merged["FutureAmount_x"]+ merged["Past Value"]

## Sort descending by FutureAmount
T_sorted = T.sort_values('FutureAmount', ascending=False)
merged_sorted = merged.sort_values("FutureAmount_x", ascending=False)
merged_sorted.rename({"FutureAmount_x": 'FutureAmount'}, axis=1, inplace=True)

