# Import necessary libraries
import numpy as np  # NumPy is used for numerical operations
from sklearn.model_selection import train_test_split  # Used to split the dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # The machine learning model we will be using
from sklearn.metrics import mean_squared_error  # Used to evaluate the performance of the model

# Create a dataset
house_sizes = np.array([750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]).reshape((-1, 1))  # Sizes of houses in square feet
house_prices = np.array([200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000])  # Corresponding prices of houses

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)  # 80% of the data is used for training, and 20% is used for testing

# Create a linear regression model
model = LinearRegression()  # Initialize the model

# Train the model using the training data
model.fit(X_train, y_train)  # The model learns the relationship between house sizes and prices from this data

# Use the trained model to make predictions on the testing data
predictions = model.predict(X_test)  # The model uses what it learned to predict house prices

# Evaluate the performance of the model
mse = mean_squared_error(y_test, predictions)  # The mean squared error is the average squared difference between the predicted and actual prices
print(f"Mean Squared Error: {mse}")  # Print the mean squared error

# pip install scikit-learn numpy