# Import necessary libraries
from sklearn import datasets  # sklearn's datasets module includes several sample datasets
from sklearn.cluster import KMeans  # The machine learning model we will be using
import matplotlib.pyplot as plt  # Used for plotting the data

# Load the iris dataset
iris = datasets.load_iris()  # The iris dataset is a classic dataset in machine learning and statistics

# Create a KMeans model
model = KMeans(n_clusters=3)  # Initialize the model. The parameter n_clusters is the number of clusters to form

# Fit the model to the iris data and predict the clusters
iris_clusters = model.fit_predict(iris.data)  # The model learns the clusters of the iris data

# Plot the clustered data
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris_clusters)  # The scatter plot shows the data points colored by their assigned cluster
plt.show()  # Display the plot

# pip install scikit-learn matplotlib
