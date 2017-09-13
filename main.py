# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:06:58 2017

@author: Milos
"""

import numpy as np
import pandas as pd

from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

import scipy.sparse as sp
from scipy.sparse.linalg import svds

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'movie':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
    
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

if __name__ == "__main__":
    header = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv("dataset/u.data", sep='\t', names=header)
    train_data, test_data = cv.train_test_split(df, test_size=0.25)
    userCount = df.user_id.unique().shape[0]
    movieCount = df.movie_id.unique().shape[0]
    
    #Memory-Based Collaborative Filtering
    train_data_matrix = np.zeros((userCount, movieCount))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]
    test_data_matrix = np.zeros((userCount, movieCount))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    movie_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

    movie_prediction = predict(train_data_matrix, movie_similarity, type='movie')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print('Movie-based CF RMSE: ' + str(rmse(movie_prediction, test_data_matrix)))
    
    # Model based Collaborative Filtering
    sparsity=round(1.0-len(df)/float(userCount*movieCount),3)
    print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')
    
    #Dobijanje SVD (singular value decomposition) komponente iz trenirate matrice
    u, s, vt = svds(train_data_matrix, k = 20)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    print('User-based CF RMSE: ' + str(rmse(X_pred, test_data_matrix)))