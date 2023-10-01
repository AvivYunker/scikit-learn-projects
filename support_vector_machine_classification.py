# Import necessary libraries
from sklearn import datasets  # sklearn's datasets module includes several sample datasets
from sklearn.model_selection import train_test_split  # Used to split the dataset into training and testing sets
from sklearn import svm  # The machine learning model we will be using
from sklearn.metrics import accuracy_score  # Used to evaluate the performance of the model

# Load the iris dataset
iris = datasets.load_iris()  # The iris dataset is a classic dataset in machine learning and statistics

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)  # 80% of the data is used for training, and 20% is used for testing

# Create a SVM classifier
model = svm.SVC()  # Initialize the model

# Train the model using the training data
model.fit(X_train, y_train)  # The model learns the relationship between the iris measurements and their species from this data

# Use the trained model to make predictions on the testing data
predictions = model.predict(X_test)  # The model uses what it learned to predict iris species

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, predictions)  # The accuracy is the proportion of correct predictions
print(f"Accuracy: {accuracy}")  # Print the accuracy

# pip install scikit-learn numpy