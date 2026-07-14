# Dataset Documentation
## Dataset Name
Bank Marketing Dataset

## Source
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/bank+marketing

## Description
This dataset contains information collected during direct marketing campaigns conducted by a Portuguese banking institution.
The objective is to predict whether a customer will subscribe to a term deposit after being contacted during a telemarketing campaign.

## Files
### bank-additional-full.csv
This is the primary dataset used by the application.
Contains all customer records together with demographic information, campaign information and economic indicators.

## Dataset Characteristics
- Number of Records: 41, 188
- Number of Features: 20 Input Var iables
- Target Variable: y
- Learning Task: Binary Classification

## Feature Categories
### Customer Information
- age
- Job
- marital
- education
### Financial Information
- default
- housing
- loan
### Campaign Inforation
- contact
- month
- day_of_week
- duration
- campaign
- pdays
- previous
- poutcome
### Economic Indicators
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

## Target Variable
y

- yes - Customer subscribed
- no -> Customer did not subscribe
Within this application:
- 1 = Subscribed
- 0 = Not Subscr ibed

## Purpose
This dataset is used for:
- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Machine Learning Classification
- Business Recommendation Generation

## Citation
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

Available at: 
    [pdf] http://dx.doi.org/10.1016/j.dss.2014.03.001
    [bib] http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt