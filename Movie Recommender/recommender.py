import numpy as np
import pandas as pd
column_names=['user_id','item_id','rating','timestamp']
data=pd.read_csv('movie recommender\\ratings.csv',)
data.head()

#PREVIEW
![1](https://user-images.githubusercontent.com/91963900/178301808-55316a8c-85d4-4792-952c-fdb5776aea8d.PNG)

movies=pd.read_csv('movie recommender\\movies.csv')
movies.head()
#PREVIEW
![2](https://user-images.githubusercontent.com/91963900/178302157-a3a1b376-bc46-412e-ae54-3f3631713a75.PNG)

df=data.merge(movies,on='movieId')

n_users=df['userId'].nunique()
n_items=df['movieId'].nunique()
#print('Number of users: '+str(n_users))
#print('Number of items: '+str(n_items))

from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(df, test_size=0.25)
train_data_matrix=data.pivot('userId','movieId','rating').fillna(0).to_numpy()
test_data_matrix=data.pivot('userId','movieId','rating').fillna(0).to_numpy()
from sklearn.metrics.pairwise import pairwise_distances
user_similarity=pairwise_distances(train_data_matrix,metric='cosine')
item_similarity=pairwise_distances(train_data_matrix.T,metric='cosine')
def Predict(ratings, similarity, type='user'):
    if type=='user':
        mean_user_rating=ratings.mean(axis=1)
        rating_diff=(ratings-mean_user_rating[:,np.newaxis])
        pred=mean_user_rating[:,np.newaxis]+similarity.dot(rating_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
    elif type=='item':
        pred=ratings.dot(similarity)/np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction=Predict(train_data_matrix,item_similarity,type='item')
user_prediction=Predict(train_data_matrix,user_similarity,type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction,ground_truth):
    prediction=prediction[ground_truth.nonzero()].flatten()
    ground_truth=ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,ground_truth))

#print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
#print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

class CfRec():
    def __init__(self, M, X, items, k=20, top_n=10):
        self.X = X
        self.M = M
        self.k = k
        self.top_n = top_n
        self.items = items
        
    def recommend_user_based(self, user):
        rec_items=self.X
        seen_mask = self.M.loc[user].gt(0)
        seen = seen_mask[seen_mask==True].index.to_list()
        rec_items = rec_items.drop(seen,axis=1).loc[user].sort_values(ascending=False).head(self.top_n)
        # return recommendations - top similar users rated movies
        return (rec_items.index.to_frame()
                                .reset_index(drop=True)
                                .merge(self.items))

    def recommend_item_based(self, item):
        liked = self.items.loc[self.items.movieId.eq(item), 'title'].item()
        print(f"Because you liked {liked}, we'd recommend you to watch:")
        # get index of movie
        ix = self.M.columns.get_loc(item)
        # Use it to index the Item similarity matrix
        i_sim = self.X[ix]
        # obtain the indices of the top k most similar items
        most_similar = self.M.columns[i_sim.argpartition(-(self.k+1))[-(self.k+1):]]
        return (most_similar.difference([item])
                                 .to_frame()
                                 .reset_index(drop=True)
                                 .merge(self.items)
                                 .head(self.top_n))
def because_user_liked(user_item_m, movies, ratings, user):
    ix_user_seen = user_item_m.loc[user]>0.
    seen_by_user = user_item_m.columns[ix_user_seen]
    return (seen_by_user.to_frame()
                 .reset_index(drop=True)
                 .merge(movies)
                 .assign(userId=user)
                 .merge(ratings[ratings.userId.eq(user)])
                 .sort_values('rating', ascending=False).head(10))

user_item_m=data.pivot('userId','movieId','rating').fillna(0)
similar=pd.DataFrame(index=user_item_m.index,columns=user_item_m.columns,data=item_prediction)
rec=CfRec(user_item_m,similar,movies)
because_user_liked(user_item_m,movies,data,68)
#PREVIEW
![3](https://user-images.githubusercontent.com/91963900/178303312-d06bf7ad-4e6b-4334-a09c-6a54f81b1a02.PNG)

rec.recommend_user_based(68)
#PREVIEW
![4](https://user-images.githubusercontent.com/91963900/178303567-d2f43eff-8d41-4110-8073-2fc1f9cc3834.PNG)

rec=CfRec(user_item_m,item_similarity,movies)
rec.recommend_item_based(1)
#PREVIEW
![5](https://user-images.githubusercontent.com/91963900/178304029-963cd74a-38c2-4689-a2a3-62426b8711e7.PNG)

sparsity=round(1.0 - len(df)/float(n_items*n_users),3)
import scipy.sparse as sp
from scipy.sparse.linalg import svds
u ,s ,vt = svds(train_data_matrix,k=20)
s_diag_matrix=np.diag(s)
X_pred=np.dot(np.dot(u ,s_diag_matrix), vt)
print('User-based CF MSE: '+str(rmse(X_pred,train_data_matrix)))
#OUTPUT
![6](https://user-images.githubusercontent.com/91963900/178304484-d1a22ad7-1513-4c28-8b15-2228ecbe73d8.PNG)
