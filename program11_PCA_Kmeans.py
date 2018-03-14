
# PCA, Principal Component Analysis
# dimensionality reduction

# we want to find the direction in which the variance is the largest
# we find the maximum-variance direction

# We use matrices and transformations
# we can use 2 dimensions to visualize our transformation

# variance, sigma_squred = (1/M) \times \sum (x - \mu)^2
# where M data points

# covariance matrix, how does one feature vary as another feature varies
# we find the eigenvectors of the covariance matrix

# the diagonals of the covariance matrix equal to 1

# (1) compute the covariance matrix
# (2) eigenvalues and eigenvectors of the covariance matrix
# (3) keep the eigenvectors with the largest eigenvalues



# we now implement PCA
# we do dimensionality reduction

# we import libraries

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# we use sklearn to load datasets
from sklearn import datasets



# we load the Iris dataset from sklearn
data = datasets.load_iris()

#print(data)

# we now define the variables X and Y, our data
X = data.data
Y = data.target

m = X.shape[0]



# function to normalise data
def normalise(x):
    x_std = x - np.mean(x, axis=0)

    # we use np.std(.) to compute the standard deviation
    x_std = np.divide(x_std, np.std(x_std, axis=0))

    return x_std



def decompose(x):
    cov = np.matmul(x.T, x)

    print('\n Covariance matrix')
    print(cov)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    print('\n Eigenvectors')
    print(eig_vecs)

    print('\n Eigenvalues')
    print(eig_vals)

    return eig_vals, eig_vecs, cov



# we now find which eigenvectors are important
def whicheigs(eig_vals):
    total = sum(eig_vals)

    # we use descending order, we use "sorted(eig_vals, reverse=True)"

    # we define the variance percentage
    var_percent = [(i/total)*100 for i in sorted(eig_vals, reverse=True)]

    cum_var_percent = np.cumsum(var_percent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Variance along different principal components')

    ax.grid()

    plt.xlabel('Principal component')
    plt.ylabel('Percentage total variance accounted for')

    ax.plot(cum_var_percent, '-ro')

    ax.bar(range(len(eig_vals)), var_percent)

    plt.xticks(np.arange(len(eig_vals)), ('PC{}'.format(i) for i in range(len(eig_vals))))

    plt.show()



# we now call the functions
X_std = normalise(X)

eig_vals, eig_vecs, cov = decompose(X_std)

whicheigs(eig_vals)



def reduce(x, eig_vecs, dims):
    W = eig_vecs[:, :dims]

    print('\n Dimension reducing matrix')
    print(W)

    return np.matmul(x,W), W



colour_dict = {0:'r', 1:'g', 2:'b'}

colour_list = [colour_dict[i] for i in list(Y)]

def plotreduced(x):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # "x[:,0]" is the first principal component
    #ax.scatter(x[:,0], x[:,1], x[:,2])

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colour_list)

    plt.grid()
    plt.show()



# we now call the functions
X_std = normalise(X)

eig_vals, eig_vecs, cov = decompose(X_std)

whicheigs(eig_vals)

#X_reduced, transform = reduce(X_std, eig_vecs, 3)

dim = 3
#dim = 1
X_reduced, transform = reduce(X_std, eig_vecs, dim)

# we plot the graph with the reduced data
plotreduced(X_reduced)



epochs = 10

def k_means(x, y, centroids=3):

    # we use 3 dimensions
    positions = 2*np.random.rand(centroids, 3) - 1

    m = x.shape[0]

    # for each epoch
    for i in range(epochs):
        assignments = np.zeros(m)

        # for each point in the data
        for datapoint in range(m):

            # compute the difference between centroid and datapoint
            difference = X_reduced[datapoint] - positions

            # we use the Euclidean distance
            norms = np.linalg.norm(difference, 2, axis=1)

            assignment = np.argmin(norms)

            assignments[datapoint] = assignment

        # for each centroid
        for c in range(centroids):
            positions[c] = np.mean(x[assignments == c])

    print('\n Assignments')
    print(assignments)

    print('\n Labels')
    print(Y)

    # print the positions of the centroids
    print(positions)

    return positions



k_means(X_reduced, Y, 3)





