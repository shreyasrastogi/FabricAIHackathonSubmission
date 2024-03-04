# Synapse Analytics notebook source

# METADATA ********************

# META {
# META   "synapse": {
# META     "lakehouse": {
# META       "default_lakehouse": "139dd0af-8c39-4ba7-96ec-db2946ced9e7",
# META       "default_lakehouse_name": "EconomicDataLakehouse",
# META       "default_lakehouse_workspace_id": "4abc1a44-3ea3-432b-a6ab-b7f7024df9c9",
# META       "known_lakehouses": [
# META         {
# META           "id": "139dd0af-8c39-4ba7-96ec-db2946ced9e7"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# 
# #### Run the cell below to install the required packages for Copilot


# CELL ********************


#Run this cell to install the required packages for Copilot
%pip install https://aka.ms/chat-magics-0.0.0-py3-none-any.whl
%load_ext chat_magics


# CELL ********************

# ATTENTION: AI-generated code can include errors or operations you didn't intend. Review the code in this cell carefully before running it.


companies_df = spark.sql(f"SELECT * FROM economicdatalakehouse.companies")


# CELL ********************

# MAGIC %%code
# MAGIC Load economic_data into DataFrame

# CELL ********************

df = spark.table('economicdatalakehouse.economic_data')

# CELL ********************

df_economic_data = spark.table('economicdatalakehouse.economic_data')

# CELL ********************

# MAGIC %%code
# MAGIC Load purchase into DataFrame

# CELL ********************

df_purchase = spark.table('economicdatalakehouse.purchase')

# CELL ********************

# MAGIC %%code
# MAGIC display graphical trend of mortgage_rate_30y based on quaterandyear order by year,quarter

# CELL ********************

import matplotlib.pyplot as plt

# Sort the DataFrame by year and quarter
sorted_df = df.select("quarterandyear", "mortgage_rate_30y").orderBy("year", "quarter")

# Extract the quarter and year values
quarters = sorted_df.select("quarterandyear").rdd.flatMap(lambda x: x).collect()
rates = sorted_df.select("mortgage_rate_30y").rdd.flatMap(lambda x: x).collect()

# Plot the trend
plt.plot(quarters, rates)
plt.xlabel("Quarter and Year")
plt.ylabel("Mortgage Rate 30Y")
plt.title("Trend of Mortgage Rate 30Y")
plt.xticks(rotation=45)
plt.show()

# CELL ********************

# MAGIC %%code
# MAGIC display graphical trend of cpi based on quaterandyear order by year,quarter

# CELL ********************

import matplotlib.pyplot as plt

# Sort the DataFrame by year and quarter
sorted_df = df.select("quarterandyear", "cpi").orderBy("year", "quarter")

# Extract the quarter and year values
quarters = sorted_df.select("quarterandyear").rdd.flatMap(lambda x: x).collect()
cpi = sorted_df.select("cpi").rdd.flatMap(lambda x: x).collect()

# Plot the trend
plt.plot(quarters, cpi)
plt.xlabel("Quarter and Year")
plt.ylabel("CPI")
plt.title("Trend of CPI")
plt.xticks(rotation=45)
plt.show()

# CELL ********************

# MAGIC %%code
# MAGIC display graphical trend of unemployment_rate based on quaterandyear order by year,quarter

# CELL ********************

import matplotlib.pyplot as plt

# Sort the DataFrame by year and quarter
sorted_df = df.select("quarterandyear", "unemployment_rate").orderBy("year", "quarter")

# Extract the quarter and year values
quarters = sorted_df.select("quarterandyear").rdd.flatMap(lambda x: x).collect()
unemployment_rate = sorted_df.select("unemployment_rate").rdd.flatMap(lambda x: x).collect()

# Plot the trend
plt.plot(quarters, unemployment_rate)
plt.xlabel("Quarter and Year")
plt.ylabel("Unemployment Rate")
plt.title("Trend of Unemployment Rate")
plt.xticks(rotation=45)
plt.show()

# CELL ********************

# MAGIC %%code
# MAGIC display graphical trend of gdp_value based on quaterandyear order by year,quarter

# CELL ********************

import matplotlib.pyplot as plt

# Sort the DataFrame by year and quarter
sorted_df = df.select("quarterandyear", "gdp_value").orderBy("year", "quarter")

# Extract the quarter and year values
quarters = sorted_df.select("quarterandyear").rdd.flatMap(lambda x: x).collect()
gdp_value = sorted_df.select("gdp_value").rdd.flatMap(lambda x: x).collect()

# Plot the trend
plt.plot(quarters, gdp_value)
plt.xlabel("Quarter and Year")
plt.ylabel("GDP Value")
plt.title("Trend of GDP Value")
plt.xticks(rotation=45)
plt.show()

# CELL ********************

# MAGIC %%code
# MAGIC relation between gdp_value and unemployment_rate

# CELL ********************

import pandas as pd

# Load economic_data into a DataFrame
df = spark.table('economicdatalakehouse.economic_data')

# Select the columns of interest
data = df.select("gdp_value", "unemployment_rate").toPandas()

# Calculate the correlation between GDP value and unemployment rate
correlation = data["gdp_value"].corr(data["unemployment_rate"])

correlation

# CELL ********************

# MAGIC %%code
# MAGIC show graphical trend between gdp_value and unemployment_rate with gdp_value as X and unemployment_rate as Y and order by gdp_value

# CELL ********************

import matplotlib.pyplot as plt

# Sort the DataFrame by gdp_value
sorted_df = df.select("gdp_value", "unemployment_rate").orderBy("gdp_value")

# Extract the gdp_value and unemployment_rate values
gdp_value = sorted_df.select("gdp_value").rdd.flatMap(lambda x: x).collect()
unemployment_rate = sorted_df.select("unemployment_rate").rdd.flatMap(lambda x: x).collect()

# Plot the trend
plt.plot(gdp_value, unemployment_rate)
plt.xlabel("GDP Value")
plt.ylabel("Unemployment Rate")
plt.title("Trend of Unemployment Rate based on GDP Value")
plt.show()

# CELL ********************

# MAGIC %%code
# MAGIC show graphical trend between gdp_value and cpi with gdp_value as X and cpi as Y and order by gdp_value


# CELL ********************

import matplotlib.pyplot as plt

# Sort the DataFrame by gdp_value
sorted_df = df.select("gdp_value", "cpi").orderBy("gdp_value")

# Extract the gdp_value and cpi values
gdp_value = sorted_df.select("gdp_value").rdd.flatMap(lambda x: x).collect()
cpi = sorted_df.select("cpi").rdd.flatMap(lambda x: x).collect()

# Plot the trend
plt.plot(gdp_value, cpi)
plt.xlabel("GDP Value")
plt.ylabel("CPI")
plt.title("Trend of CPI based on GDP Value")
plt.show()

# CELL ********************

# MAGIC %%code
# MAGIC Create a model to predict mortgage_rate_30y for future quaterandyear based on current data

# CELL ********************

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load economic_data into a DataFrame
df = spark.table('economicdatalakehouse.economic_data')

# Select the features and target variable
data = df.select("quarterandyear", "mortgage_rate_30y").toPandas()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["quarterandyear"], data["mortgage_rate_30y"], test_size=0.2, random_state=42)

# Convert quarterandyear to numerical representation using label encoding
X_train_encoded = pd.factorize(X_train)[0].reshape(-1, 1)
X_test_encoded = pd.factorize(X_test)[0].reshape(-1, 1)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_encoded, y_train)

# Predict mortgage_rate_30y for future quarterandyear values
future_quarterandyear = ["Q3-2022", "Q4-2022", "Q1-2023"]
future_quarterandyear_encoded = pd.factorize(future_quarterandyear)[0].reshape(-1, 1)
predicted_rates = model.predict(future_quarterandyear_encoded)

predicted_rates

# CELL ********************

# MAGIC %%code
# MAGIC show above prediction with current data in a graphical way

# CELL ********************

import matplotlib.pyplot as plt

# Plot the current data
plt.plot(data["quarterandyear"], data["mortgage_rate_30y"], label="Current Data")

# Plot the predicted rates
plt.plot(future_quarterandyear, predicted_rates, 'ro', label="Predicted Rates")

plt.xlabel("Quarter and Year")
plt.ylabel("Mortgage Rate 30Y")
plt.title("Predicted Mortgage Rate 30Y for Future Quarter and Year")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# CELL ********************

# MAGIC %%code 
# MAGIC create a new table by joining economic_data and purchase table 
# MAGIC join on columns quarterandyear,quarter,year
# MAGIC sum revenue based on productcategory
# MAGIC table should have columns  quarterandyear,quarter,year,company_id,productcategory,revenue

# CELL ********************

df_join = economic_data.join(purchase, 
                             (economic_data.quarterandyear == purchase.quarterandyear) & 
                             (economic_data.quarter == purchase.quarter) & 
                             (economic_data.year == purchase.year), 
                             "inner")

df_new_table = df_join.groupBy("quarterandyear", "quarter", "year", "company_id", "productcategory") \
                      .agg({"revenue": "sum"}) \
                      .withColumnRenamed("sum(revenue)", "revenue")

df_new_table.show()

# MARKDOWN ********************

# **updated above code to include dataframes created above to resolve error
# 
# used %%fix_errors remove duplicate columns in join to fix further error**

# CELL ********************

df_join = df_economic_data.join(df_purchase, 
                             (df_economic_data["quarterandyear"] == df_purchase["quarterandyear"]) & 
                             (df_economic_data["quarter"] == df_purchase["quarter"]) & 
                             (df_economic_data["year"] == df_purchase["year"]), 
                             "inner")

df_new_table = df_join.drop(df_purchase["quarterandyear"], df_purchase["quarter"], df_purchase["year"]) \
                      .groupBy("quarterandyear", "quarter", "year", "company_id", "productcategory") \
                      .agg({"revenue": "sum"}) \
                      .withColumnRenamed("sum(revenue)", "revenue")

df_new_table.show()

# CELL ********************

# MAGIC %%code 
# MAGIC add a new column in df_new_table as company_name
# MAGIC  with company_name from companies table based on company_id

# CELL ********************




# CELL ********************

df_new_table_with_company_name = df_new_table.join(companies_df, "company_id", "inner").withColumnRenamed("company_name", "new_column_name")

# CELL ********************

# MAGIC %%code
# MAGIC remove columns company_type,employees,location from df_new_table_with_company_name

# CELL ********************

df_new_table_with_company_name = df_new_table_with_company_name.drop("company_type", "employees", "location")

# CELL ********************

df_new_table_with_company_name.show()

# CELL ********************

# MAGIC %%code
# MAGIC rename new_column_name as company_name in df_new_table_with_company_name

# CELL ********************

df_new_table_with_company_name = df_new_table_with_company_name.withColumnRenamed("new_column_name", "company_name")

# CELL ********************

df_new_table_with_company_name.show()

# CELL ********************

# MAGIC %%code
# MAGIC save df_new_table_with_company_name as table company_sales

# CELL ********************

df_new_table_with_company_name.write.saveAsTable("company_sales")

# CELL ********************

# MAGIC %%code
# MAGIC load company_sales into dataframe and show 

# CELL ********************

df_company_sales = spark.table("company_sales")
df_company_sales.show()

# CELL ********************

# MAGIC %%code
# MAGIC calculate total revenue from revenue for each company from company_sales 
# MAGIC and order by total revenue
# MAGIC rename sum(revenue) to Total_Revenue

# CELL ********************

df_total_revenue = df_company_sales.groupby("company_name").sum("revenue").orderBy("sum(revenue)").withColumnRenamed("sum(revenue)", "Total_Revenue")

# CELL ********************

df_total_revenue.show()

# CELL ********************

# MAGIC %%code 
# MAGIC show Total_revenue in full number

# CELL ********************

df_total_revenue.select("company_name", "Total_Revenue").show()

# CELL ********************

# MAGIC %%code
# MAGIC based on above which is the company having top revenue

# CELL ********************

df_top_company = df_total_revenue.orderBy("Total_Revenue", ascending=False).first()
top_company = df_top_company["company_name"]
top_company

# CELL ********************

# MAGIC %%code
# MAGIC what is the top selling productcategory of 'Davis Group'

# CELL ********************

df_davis_group = df_company_sales.filter(df_company_sales["company_name"] == "Davis Group")
df_top_productcategory_davis_group = df_davis_group.groupBy("productcategory").sum("revenue").orderBy("sum(revenue)", ascending=False).first()
top_productcategory_davis_group = df_top_productcategory_davis_group["productcategory"]
top_productcategory_davis_group

# CELL ********************

# MAGIC %%code
# MAGIC top selling productcategory from company_sales

# CELL ********************

df_top_productcategory = df_company_sales.groupBy("productcategory").sum("revenue").orderBy("sum(revenue)", ascending=False).first()
top_productcategory = df_top_productcategory["productcategory"]
top_productcategory

# CELL ********************

# MAGIC %%code
# MAGIC which company has top selling product category as 'Home & Kitchen'

# CELL ********************

df_home_kitchen = df_company_sales.filter(df_company_sales["productcategory"] == "Home & Kitchen")
df_top_company_home_kitchen = df_home_kitchen.groupBy("company_name").sum("revenue").orderBy("sum(revenue)", ascending=False).first()
top_company_home_kitchen = df_top_company_home_kitchen["company_name"]
top_company_home_kitchen
