# Import necessary libraries
from sklearn import datasets  # sklearn's datasets module includes several sample datasets
from sklearn.decomposition import PCA  # The machine learning model we will be using
import matplotlib.pyplot as plt  # Used for plotting the data

# Load the iris dataset
iris = datasets.load_iris()  # The iris dataset is a classic dataset in machine learning and statistics

# Create a PCA model
model = PCA(n_components=2)  # Initialize the model. The parameter n_components is the number of dimensions to reduce to

# Fit the model to the iris data and apply the dimensionality reduction
iris_reduced = model.fit_transform(iris.data)  # The model learns the principal components of the iris data and applies the dimensionality reduction

# Plot the reduced data
plt.scatter(iris_reduced[:, 0], iris_reduced[:, 1], c=iris.target)  # The scatter plot shows the data points in the reduced dimensional space, colored by species
plt.show()  # Display the plot

# pip install scikit-learn numpy matplotlib