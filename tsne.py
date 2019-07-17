import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

#load data
ppl = int(sys.argv[1])
A = np.load(sys.argv[2])
B = np.load(sys.argv[3])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(A[:,0], A[:,1], A[:,2])
ax.scatter(B[:,0], B[:,1], B[:,2])

plt.show()

#PCA
X = np.r_[A,B]
X2 = PCA(n_components=2).fit_transform(X)

A2 = X2[:A.shape[0], :]
B2 = X2[A.shape[0]:, :]
plt.scatter(A2[:,0], A2[:,1])
plt.scatter(B2[:,0], B2[:,1])

plt.show()

#TSNE
def plot_tsne(perplexity=5):
    X3 = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
    A3 = X3[:A.shape[0], :]
    B3 = X3[A.shape[0]:, :]
    plt.scatter(A3[:,0], A3[:,1])
    plt.scatter(B3[:,0], B3[:,1])
    plt.show()

plot_tsne(ppl)
