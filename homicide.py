"""
DATA SCIENCE
1. Data Munging
    a. missing data
    b. outliers
    c. data types
    d. duplicate rows
    e. untidy
    f. need to process columns
    
2. Data Analytics
    a. Using Pandas
3. Data Representation/Visualization
    a. Using Seaborn and Plotly 

RESEARCH AND STATISTICS
1. Describe and summarize the data
    a. Identify the columns used and there unique values
    b. Identify the mean for relevant columns
    c. Identify the frequency for relevant columns
    d. Show frequency using a graphical representation
2. Identify relationships between variables
    a. Correlate variables
    b. Linear regression
3. Compare variables

4. Identify the difference between variables
5. Forecast outcomes
"""
import pandas as pd
import plotly.plotly as py
import numpy as np
#import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#read csv data
homicide_reports = 'data/raw/homicide_reports.csv'
df = pd.read_csv(homicide_reports)

#create a new copy of the data
homicide_df = df.copy()
homicide_df = (homicide_df.drop(['Record ID','Agency Code','Agency Name','Agency Type',
                                 'Victim Ethnicity','Perpetrator Ethnicity','Record Source'],axis=1))
#identify number of rows and columns of the data
#print(homicide_df.shape)

#concise summary of the dataframe to check for missing values and data type of columns
#print(homicide_df.info())

#convert datatypes of columns to object
homicide_df[['Incident', 'Victim Count','Perpetrator Count']] = homicide_df[['Incident','Victim Count','Perpetrator Count']].astype(object)
#convert datatypes of columns to numerical
homicide_df[['Perpetrator Age', 'Victim Count', 'Perpetrator Count']] = homicide_df[['Perpetrator Age', 'Victim Count', 'Perpetrator Count']].apply(pd.to_numeric, errors ='ignore')
#print(homicide_df.info())
#convert datatypes of columns to category
for col in ['City','State','Year','Month','Crime Type','Crime Solved','Victim Sex','Victim Race','Perpetrator Sex','Perpetrator Race','Relationship','Weapon']:
    homicide_df[col] = homicide_df[col].astype('category', ordered = True)
#print(homicide_df.dtypes)
#prints categories
#print(homicide_df['Victim Sex'].cat.categories)
#print numerical values
#print(homicide_df['Victim Sex'].cat.codes.unique())
#change category data type columns to encoded values
"""for col in ['City','State','Year','Month','Crime Type','Crime Solved','Victim Sex','Victim Race','Perpetrator Sex','Perpetrator Race','Relationship','Weapon']:
    homicide_df[col] = homicide_df[col].cat.codes
print(homicide_df.loc[:10])"""
#check to see if any value is NaN in the dataframe
#print('are there any null values?',homicide_df.isnull().values.any())
#print('if there is string in numeric column',np.any([isinstance(val, str) for val in homicide_df['Perpetrator Age']]))
#np.isreal to check the type of each element (applymap applies a function to each element in the DataFrame)
#print(homicide_df[~homicide_df.applymap(np.isreal).all(1)])
def check_type(homicide_df,col):
    return homicide_df.loc[homicide_df[col].apply(type)==str,col]
#print('Check Type for Perpetrator Age',check_type(homicide_df, 'Perpetrator Age'))
#print(check_type(homicide_df))
#convert data types of values other than int to NaN
homicide_df['Perpetrator Age']=pd.to_numeric(homicide_df['Perpetrator Age'], errors='coerce')
#print('are there any null values?',homicide_df.isnull().values.any())
homicide_df['Perpetrator Age'] = homicide_df['Perpetrator Age'].fillna(0).astype(int)
#print('are there any null values?',homicide_df.isnull().values.any())
#print('Is there a string in numeric column?',np.any([isinstance(val, str) for val in homicide_df['Perpetrator Age']]))
#generate descriptive statistics that summarize the central tendency, dispersion 
#and shape of the dataset’s distribution, excluding NaN values.
#print(homicide_df.describe(include=['O']))

#show unique values used on the dataset
"""
print('Crime Types = ', homicide_df['Crime Type'].unique())
print('Victim Sex = ', homicide_df['Victim Sex'].unique())
print('Victim Races = ', homicide_df['Victim Race'].unique())
print('Perpetrator Races = ', homicide_df['Perpetrator Race'].unique())
print('Relationships = ', homicide_df['Relationship'].unique())
print('Weapons = ', homicide_df['Weapon'].unique())
print('Record Sources = ', homicide_df['Record Source'].unique())
"""


#before checking and dropping rows, check whether all indices are of unique values
#print('Are indices unique?',homicide_df.index.is_unique)

#check for outliers with the data
#print('1980 < Year > 2014 - ',((homicide_df['Year'] > 2014) | (homicide_df['Year'] < 1980)).any())
clean_year = homicide_df[homicide_df['Year'] < 1980]
#print(clean_year)
#oldest living person in USA recorded between 1980-2014 was 119 yrs old
#print('0 < Victim Age > 119 - ',((homicide_df['Victim Age'] > 119) | (homicide_df['Victim Age'] < 0)).any())
outlier_victim_age = homicide_df[homicide_df['Victim Age'] > 119]
#print('Outliear data:\n', outlier_victim_age['Victim Age'].head())
homicide_df = homicide_df.drop(homicide_df[((homicide_df['Victim Age'] > 119) | (homicide_df['Victim Age'] < 0))].index)
#print('0 < Victim Age > 119 - ',((homicide_df['Victim Age'] > 119) | (homicide_df['Victim Age'] < 0)).any())
#print('0 < Perpetrator Age > 119 - ',((homicide_df['Perpetrator Age'] > 119) | (homicide_df['Perpetrator Age'] < 0)).any())
#drop indices with Perpetrator Age < 8 since youngest recorded perpetrator for homicide between 1980-2014 was 8
homicide_df = homicide_df.drop(homicide_df[(homicide_df['Perpetrator Age'] < 8)].index)
#print(homicide_df.corr(method='pearson'))


#count frequency dropna=True to drop missing values
cities = homicide_df['City'].value_counts().head()
states = homicide_df['State'].value_counts()
years = homicide_df['Year'].value_counts()
months = homicide_df['Month'].value_counts()
crime_types = homicide_df['Crime Type'].value_counts()
crime_solved = homicide_df['Crime Solved'].value_counts()
victim_sex = homicide_df['Victim Sex'].value_counts()
victim_ages = homicide_df['Victim Age'].value_counts()
victim_races = homicide_df['Victim Race'].value_counts()
perpetrator_sex = homicide_df['Perpetrator Sex'].value_counts()
perpetrator_ages = homicide_df['Perpetrator Age'].value_counts()
perpetrator_races = homicide_df['Perpetrator Race'].value_counts()
relationships = homicide_df['Relationship'].value_counts()
weapons = homicide_df['Weapon'].value_counts()

#visual exploratory data analysis 
#also another way to check for outliers
def box_plot(homicide_df, column, by, rot):
    homicide_df.boxplot(column=column, by=by, rot=rot)
    plt.show()
#box_plot(homicide_df, 'Victim Age', 'Year', 90)
#box_plot(homicide_df, 'Victim Age', 'State', 90)
#box_plot(homicide_df, 'Victim Age', 'Crime Type', 0)
#box_plot(homicide_df, 'Victim Age', 'Victim Sex', 0)
#box_plot(homicide_df, 'Victim Age', 'Victim Race', 45)
#box_plot(homicide_df, 'Victim Age', 'Perpetrator Sex', 0)
#box_plot(homicide_df, 'Victim Age', 'Perpetrator Race', 45)
#box_plot(homicide_df, 'Victim Age', 'Relationship', 90)
#box_plot(homicide_df, 'Victim Age', 'Weapon', 90)

#box_plot(homicide_df, 'Perpetrator Age', 'Year', 90)
#box_plot(homicide_df, 'Perpetrator Age', 'State', 90)
#box_plot(homicide_df, 'Perpetrator Age', 'Crime Type', 0)
#box_plot(homicide_df, 'Perpetrator Age', 'Victim Sex', 0)
#box_plot(homicide_df, 'Perpetrator Age', 'Victim Race', 45)
#box_plot(homicide_df, 'Perpetrator Age', 'Perpetrator Sex', 0)
#box_plot(homicide_df, 'Perpetrator Age', 'Perpetrator Race', 45)
#box_plot(homicide_df, 'Perpetrator Age', 'Relationship', 90)
#box_plot(homicide_df, 'Perpetrator Age', 'Weapon', 90)


#COMPARE VARIABLES

#Crosstab
def crosstab(homicide_df,row, col):
    return(pd.crosstab(homicide_df[row], homicide_df[col]))

page_vage = crosstab(homicide_df,'Victim Age','Perpetrator Age')
month_solved = crosstab(homicide_df,'Month','Crime Solved')
year_month = crosstab(homicide_df,'Year','Month')
relationship_vsex = crosstab(homicide_df,'Relationship','Victim Sex')
state_solved = crosstab(homicide_df,'State','Crime Solved')
relationship_weapon = crosstab(homicide_df,'Relationship','Weapon')
weapon_psex = crosstab(homicide_df,'Weapon','Perpetrator Sex')
weapon_page = crosstab(homicide_df,'Perpetrator Age','Weapon')

def heatmap(crosstab, annot,xtick, ytick, xrot, yrot):
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(crosstab,annot=annot,xticklabels=xtick, yticklabels=ytick)
    plt.xticks(rotation=xrot)
    plt.yticks(rotation=yrot)
    return heatmap
#heatmap(p_age_v_age,False,10,10,0,0)
#heatmap(month_solved,True,True,True,0,0)
#heatmap(year_month,False,True,True,90,0)
#heatmap(relationship_vsex,True,True,True,0,0)
#heatmap(state_solved,False,True,True,0,0)
#heatmap(relationship_weapon,False,True,True,90,0)
#heatmap(weapon_psex,True,True,True,0,0)
heatmap(weapon_page,False,True,10,90,0)

"""
Linear Regression
x_axis = cause, independent variable (the thing you are changing), explanatory
y_axis = effect, dependent variable (the thing you are measuring), response
"""
dummies = pd.get_dummies(homicide_df[['Crime Type','Crime Solved', 'Victim Sex', 'Victim Race', 'Perpetrator Sex', 'Perpetrator Race', 'Relationship', 'Weapon']])
#print(dummies.corr(method='pearson'))


def plot_compare_var(x, y, data):
    ax = sns.regplot(x=x, y=y, data=data)
    plt.show(ax)
    

#sns.regplot(x='Crime Solved',y='Crime Type',data=dummies)
#ax = sns.countplot(x='Victim Sex', data=homicide_df, hue='Perpetrator Sex')

#Odds Ratio
orsex = crosstab(homicide_df, 'Victim Sex', 'Perpetrator Sex')
orsex = orsex.drop("Unknown", axis=1)
orsex = orsex.apply(lambda r: r/r.sum(), axis=1)
print(orsex)
#Null Hypothesis: There is no significant relationship between Victim Sex and Female Sex

"""
oddsratio, pvalue = stats.fisher_exact(orsex)
print('Odds ratio = ', oddsratio, 'Pvalue = ', pvalue)
"""

