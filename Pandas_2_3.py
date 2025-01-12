# # Pandas
# Read "10 minutes to Pandas": https://pandas.pydata.org/docs/user_guide/10min.html before solving the exercises.
# We will use the data set "cars_data" in the exercises below. 

# Importing Pandas. 
import pandas as pd

# ### Explain what a CSV file is.
## Answer : 
##


# ### Load the data set "cars_data" through Pandas. 

# When reading in the data, either you have the data file in the same folder as your python script
# or in a seperate folder.

# Code below can be ran if you have the data file in the same folder as the script
cars = pd.read_csv("cars_data.csv")

# Code below can be ran if you have the data file in another script. 
# Notice, you must change the path according to where you have the data in your computer. 
# pd.read_csv(r'C:\Users\Antonio Prgomet\Documents\03_nbi_yh\korta_yh_kurser\python_för_ai\kunskapskontroll_1\cars_data.csv')

# ### Print the first 10 rows of the data. 
print(cars.head(10))

# ### Print the last 5 rows. 
print(cars.tail(5))

# ### By using the info method, check how many non-null rows each column have. 
cars.info()
print(cars.notnull().sum())

# ### If any column has a missing value, drop the entire row. Notice, the operation should be inplace meaning you change the dataframe itself.
cars.dropna(inplace=True)

# ### Calculate the mean of each numeric column. 
mean_cars = cars.select_dtypes(include=['number']).mean()
print(mean_cars)

# ### Select the rows where the column "company" is equal to 'honda'. 
honda_cars = cars[cars['company'] == 'honda']
print(honda_cars)

# ### Sort the data set by price in descending order. This should *not* be an inplace operation. 
desc_sorted_cars = cars.sort_values(by='price', ascending=False)
print(desc_sorted_cars)

# ### Select the rows where the column "company" is equal to any of the values (audi, bmw, porsche).
selected_car_companies = cars[cars['company'].isin(['audi', 'bmw', 'porsche'])]
print(selected_car_companies)

# ### Find the number of cars (rows) for each company. 
cars_per_company = cars.groupby('company').size()
print(cars_per_company)

# ### Find the maximum price for each company. 
price_maximum_company = cars.groupby('company')['price'].max()
print(price_maximum_company)




