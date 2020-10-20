# -*- coding: utf-8 -*-
"""
Created on Sat May 18 01:21:33 2019

@author: Dell
"""

############################ MEMORY BASED ####################################



#  CORRELATION

import pandas as pd
#INPUT DATA

column_names = ['user_id', 'item_id', 'rating', 'timestamp'] 
df=pd.read_csv("C:/Users/Dell/Desktop/dataset/file.tsv", sep='\t', names=column_names)
movie_titles=pd.read_csv("C:/Users/Dell/Desktop/dataset/Movie_Id_Titles (1).csv")
df.head()
data = pd.merge(df, movie_titles, on='item_id') 
data.head() 

#CALC. MEAN RATING OF ALL THE BOOKS AND CALC. COUNT RATING (DESCENDING ORDER)
data.groupby('title')['rating'].mean().sort_values(ascending=False).head() 
data.groupby('title')['rating'].count().sort_values(ascending=False).head() 

#DATAFRAME CREATED 
ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count()) 
#print ratings.head()
book = data.pivot_table(index ='user_id', columns ='title', values ='rating') 
l1=ratings.sort_values('num of ratings', ascending = False).head(10) 
#print l1

#CORRELATION WIITH SIMILAR BOOKS rated out below
book1_user_ratings = book['Northwest Wines and Wineries'] 
book1_user_ratings.head() 
similar_to_book1 =book .corrwith(book1_user_ratings) 
corr_book1 = pd.DataFrame(similar_to_book1, columns =['Correlation']) 
corr_book1.dropna(inplace = True) 
corr_book1.head() 
corr_book1.sort_values('Correlation', ascending = False).head(10) 
corr_book1 = corr_book1.join(ratings['num of ratings']) 
corr_book1.head() 
print corr_book1[corr_book1['num of ratings']>100].sort_values('Correlation', ascending = False).head() 



################################# MODEL BASED #################################


import numpy as np
class MF():

    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.
    #alpha=learning rate beta=regularization parameter
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and movie-feature matrix 
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
        for j in range(self.num_items)
        if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
        training_process.append((i, mse))
        if (i+1) % 20 == 0:
            print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_matrix(self):
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)
    


R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
            
            
mf= MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
training_process = mf.train()
print R
print("P x Q:")
print(mf.full_matrix())



