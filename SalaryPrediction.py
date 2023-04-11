import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

df = pd.read_csv('survey_results_public.csv' )
print(df.head())

df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
print(df.head())

df = df[df['Salary'].notnull()]
# print(df.head())

print(df.info())

df = df.dropna()  # drop the data which doesn't have a number
print(df.isnull().sum())  # we use only those examples with a valid data

df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)  # we don't want this column for training
print(df.info())

print(df['Country'].value_counts())


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'other'
    return categorical_map


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
print(df['Country'].value_counts())

df = df[df['Salary'] <= 250000]
df = df[df['Salary'] >= 10000]
df = df[df['Country'] != 'other']

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
# plt.show()

print(df['YearsCodePro'].unique())


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

print(df['EdLevel'].unique())

# transforming strings into numbers using preprocessing
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
print(df['EdLevel'].unique())

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
print(df['Country'].unique())

x = df.drop("Salary", axis=1)
y = df["Salary"]

# MAKING MODEL
linear_reg = LinearRegression()
linear_reg.fit(x, y)

y_pred = linear_reg.predict(x)

error = np.sqrt(mean_squared_error(y, y_pred))
print(error)

dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(x, y.values)

y_pred = dec_tree_reg.predict(x)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(x, y.values)
y_pred = random_forest_reg.predict(x)

error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}
regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(x, y.values)

regressor = gs.best_estimator_
regressor.fit(x, y.values)
y_pred = regressor.predict(x)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

# print(x)

# Country, edLevel, YearsCodePro
x = np.array([["United States", 'Masters degree', '15']])
print(x)

x = np.array([[13., 2., 15.]])
print(x)

y_pred = regressor.predict(x)
print(y_pred)

# Saving the file

data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

y_pred = regressor_loaded.predict(x)
print(y_pred)

