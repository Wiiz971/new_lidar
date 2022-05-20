# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:00:19 2022

@author: Vincent
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from numpy.linalg import norm
from sklearn.decomposition import PCA

rng = np.random.RandomState(401)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');
plt.show()

pca = PCA(n_components=2)
pca.fit(X)

def draw_vector(v0, v1, ax = None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth = 1,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()

def bbox_2D(X):
    '''
    Finding the four corners of an oriented bounding box (OBB)
    Input: 
         X:  points in 2-D, (num, 2)
    Output:
         cornors_rot: four corners of OBB
    '''
    # Carry out the principle component analysis
    pca = PCA(n_components = 2)
    pca.fit(X)
    x_axis = pca.components_[0]  # We follow the convention of denoting the larger eigenvector as x
    x_axis = x_axis if np.dot(x_axis, [1, 0]) > 0 else - x_axis # reverse the orientation if necessary
    x_axis = x_axis /norm(x_axis)
    
    y_axis = pca.components_[1] # We follow the convention of denoting the smaller eigenvector as y
    y_axis = y_axis if np.dot(y_axis, [0, 1]) > 0 else - y_axis # reverse the orientation if necessary
    y_axis = y_axis /norm(y_axis)
    
    
    # Find the centroid of the cluster
    C = np.mean(X, axis=0 )  
    X = X - C  
    
    # Calculate each point's projection on the two PCA components
    dist_x = np.dot(X, x_axis)
    dist_y = np.dot(X, y_axis)
    
    # Calculate the four conners in the original coordinate system
    x1, x2, y1, y2 = dist_x.max(), dist_x.min(), dist_y.max(), dist_y.min()
    cornors = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]] + C
    
    # Determine the four conners in the PCA axes
    s_base = [1, 0]
    a = np.dot(x_axis, s_base)/norm(s_base) 
    theta = np.arccos(np.clip(a, -1, 1))  # Find the roation angle
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) # Construct rotation matrix
    # Cornor points in the PCA coordinates
    cornors_rot = [ np.dot(rot_mat, np.array(x).T) for x in cornors] 
    return cornors_rot
rng = np.random.RandomState(900)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1], alpha = 0.6)
plt.axis('equal');
pca = PCA(n_components=2)
pca.fit(X)

for length, vector in zip(pca.explained_variance_, pca.components_):
    print('length: ', length, 'vector: ', vector)
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

cornors_rot = bbox_2D(X)    



for i  in  range(4):
    cor1, cor2 = cornors_rot[i % 4], cornors_rot[(i+1) % 4]
    plt.plot([cor1[0], cor2[0]], [cor1[1], cor2[1]], 'r-')
plt.axis('equal')
plt.show()

