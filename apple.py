import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print('This is the Project to predict the characteristics of an Apple by taking its size and weight as inputs.')

# here i taken the dataset which has to be examined
ds = pd.read_csv("C:\\Users\\rakes\\OneDrive\\Desktop\\apple.csv")

# here i defined the parameters or primary keys and standarized the inputs
features = ["size", "weight"]
size_mean = ds["size"].mean()
size_std = ds["size"].std()
weight_mean = ds["weight"].mean()
weight_std = ds["weight"].std()


#function to standardize the inputs 
def standardize(value, mean, std):
    return (value - mean) / std

# Input new size and weight values were taken
new_size = float(input("Enter the size in centimeters: "))
new_weight = float(input("Enter the weight in kilograms: "))

# Standardizing the input values using the mean and std from the dataset
standardized_size = standardize(new_size, size_mean, size_std)
standardized_weight = standardize(new_weight, weight_mean, weight_std)

#writing the common model training and prediction code
def predict_attribute(attribute, size, weight):
    X = ds[features]
    Y = ds[attribute]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    #model was trained successfully
    Y_pred = model.predict(X_test)
    print(f'\n{attribute.capitalize()}')
    new_data = pd.DataFrame({'size': [size], 'weight': [weight]})
    predicted_value = model.predict(new_data)
    return predicted_value[0]

# Predicting various attributes
sweetness = predict_attribute("sweetness", standardized_size, standardized_weight)
print(f'Predicted sweetness: {sweetness}')

crunchiness = predict_attribute("crunchiness", standardized_size, standardized_weight)
print(f'Predicted crunchiness: {crunchiness}')

juiciness = predict_attribute("juiciness", standardized_size, standardized_weight)
print(f'Predicted juiciness: {juiciness}')

ripeness = predict_attribute("ripeness", standardized_size, standardized_weight)
print(f'Predicted ripeness: {ripeness}')

acidity = predict_attribute("acidity", standardized_size, standardized_weight)
print(f'Predicted acidity: {acidity}')

print('\nQuality of the apple')
ds['quality'] = ds['quality'].map({'good': 100, 'bad': 0})
ds['quality'].fillna(0, inplace=True)
X = ds[features]
Y = ds["quality"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
new_data = pd.DataFrame({'size': [standardized_size], 'weight': [standardized_weight]})
predicted_quality = model.predict(new_data)

print('\n')
print(f'Predicted quality: {round(predicted_quality[0], 2)}%')