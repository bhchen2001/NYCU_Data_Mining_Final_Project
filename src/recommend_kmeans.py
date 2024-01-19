import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

from reader_df_builder import ReaderDFBuilder

class bookKMeans():
    """
    Constructor
    Input:
        data_path: 
            type: string
            description: the path of the data folder
        k:
            type: int
            description: the number of nearest neighbors
        recommend_num:
            type: int
            description: the number of recommended books
    """
    def __init__(self, data_path, cluster_num = 10, user_num = 10, recommend_num = 10):
        self.data_path = data_path
        self.reader_list = None
        self.cluster_num = cluster_num
        self.user_num = user_num
        self.recommend_num = recommend_num

    """
    Read reader id file
    """
    def load_data(self):
        self.reader_list = pd.read_csv(self.data_path + 'validate_reader_id.csv')

    """
    Recommend Books Depend on the Similar Users
        the recommended books are sorted by the occurence and average score
        user_weight = 1 / (distance)
    Input:
        similar_users:
            type: list
            description: the list of similar users
            example: [(user_id, distance), ...]
        train_bought_list:
            type: dict
            description: the buying list of train readers
            example: {user_id: ([book_id, ...], [rating_score, ...]), ...}
        test_bought_list:
            type: dict
            description: the buying list of test readers
            example: {user_id: ([book_id, ...], [rating_score, ...]), ...}
    Output:
        [book_id, ...]:
            type: list
            description: the list of recommended books
    """
    def recommend_books(self, similar_users, train_bought_list, test_bought_list):
        # for each similar user, find the books that he/she read
        # and calculate the occurence, total score and average score of each book
        recommend_book_dict = {}
        for user_idx, user in enumerate(similar_users):
            user_id, distance = user[0], user[1]
            user_weight = 1 / (distance)
            user_buying_list = train_bought_list[user_id]
            for book_idx, book in enumerate(user_buying_list[0]):
                if book in test_bought_list:
                    # the test user has already bought this book
                    continue
                if book not in recommend_book_dict:
                    recommend_book_dict[book] = {
                        'occurence': 1,
                        # the score also consider the distance between users
                        'score': user_buying_list[1][book_idx] * user_weight,
                        'avg_score': 0
                    }
                else:
                    recommend_book_dict[book]['occurence'] += 1
                    recommend_book_dict[book]['score'] += user_buying_list[1][book_idx] * user_weight
        # calculate the average score of each book
        for book_id, book in recommend_book_dict.items():
            book['avg_score'] = book['score'] / book['occurence']
        # sort the books first by occurence, then by average score
        recommend_book_list = sorted(recommend_book_dict.items(), key = lambda x: (x[1]['occurence'], x[1]['avg_score']), reverse = True)
        # return the top r books id
        return [book[0] for book in recommend_book_list[:self.recommend_num]]
    
    def user_similarity(self, user_a, user_b):
        # similarity using cosine similarity
        user_a = user_a.drop(['User_id'], axis=1)
        user_b = user_b.drop(['User_id'], axis=1)
        similarity = cosine_similarity(user_a.values.reshape(1, -1), user_b.values.reshape(1, -1))
        return similarity[0][0]

        # using euclidean distance
        # return np.linalg.norm(user_a - user_b)

    def plot_pca(self):

        reader_df_builder = ReaderDFBuilder(self.data_path)
        reader_df_builder.process()
        train_reader_df, test_reader_df, train_bought_list, test_buying_list, test_bought_list = \
            reader_df_builder.build_reader_df(self.reader_list['User_id'].tolist(), [])
        
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(train_reader_df.drop(['User_id'], axis=1))

        model = KMeans(n_clusters = self.cluster_num, random_state=0)
        model.fit(train_reader_df.drop(['User_id'], axis=1).to_numpy().tolist())
        labels = model.labels_

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1],marker='o')
        plt.title('User Dsitribution with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()    
    
    """
    MRR Criteria
        for each target user, find the first book that he/she bought in the recommended books
    Input:
        target_book_list:
            type: list
            description: the list of target books
            example: [book_id, ...]
        recommended_books:
            type: list
            description: the list of recommended books
            example: [book_id, ...]
    Output:
        mrr_q:
            type: float
            description: the mrr score
    """
    def mrr_criteria(self, target_book_list, recommended_books):
        mrr_q = 0
        for idx, book in enumerate(recommended_books):
            if book in target_book_list:
                mrr_q = 1 / (idx + 1)
                break
        return mrr_q
    
    """
    MAP Criteria
        for each target user, find the average precision of the recommended books
    Input:
        target_book_list:
            type: list
            description: the list of target books
            example: [book_id, ...]
        recommended_books:
            type: list
            description: the list of recommended books
            example: [book_id, ...]
    Output:
        map_aveq:
            type: float
            description: the map score
    """
    def map_criteria(self, target_book_list, recommended_books):
        hit = 0
        hit_location = []
        map_aveq = 0
        for idx, book in enumerate(recommended_books):
            if book in target_book_list:
                hit += 1
                hit_location.append(idx + 1)
                map_aveq += hit / (idx + 1)
        if hit != 0:
            map_aveq /= hit
        return map_aveq, hit_location

    """
    kMeans Algorithm
        1. split the data into training and testing
        2. build reader dataframe and buying list
        3. fit the kMeans model
        4. for each user in the testing set, find the top k similar users in the training set
            and find the books that similar users read
        5. calculate the mrr and map score for the target user
        6. output the average mrr and map score
    """
    def kMeans(self, n = 5):

        # split the dataframe into chunks with same size n
        reader_id_chunks = np.array_split(self.reader_list, n)

        # cross validation
        mrr_cv = []
        map_cv = []

        for fold_round in range(n):
            print("[CV{}/{}]".format(fold_round + 1, n))
            # take the fold_round chunk as testing set
            test = reader_id_chunks[fold_round]
            # take the rest chunks as training set
            train = []
            for i in range(n):
                if i != fold_round:
                    train.append(reader_id_chunks[i])
            train = pd.concat(train)

            # split the data into training and testing
            # train, test = train_test_split(self.reader_list, test_size=0.2, random_state=42)
            fold_reader_id_list = train['User_id'].tolist()

            # build reader dataframe and buying list
            reader_df_builder = ReaderDFBuilder(self.data_path)
            reader_df_builder.process()
            train_reader_df, test_reader_df, train_bought_list, test_buying_list, test_bought_list = \
                reader_df_builder.build_reader_df(train['User_id'].tolist(), test['User_id'].tolist())
            
            # train the kMeans model
            # model = NearestNeighbors(n_neighbors=self.k_nearest)
            # model.fit(train_reader_df.drop(['User_id'], axis=1).to_numpy().tolist())
            model = KMeans(n_clusters = self.cluster_num, random_state=0)
            model.fit(train_reader_df.drop(['User_id'], axis=1).to_numpy().tolist())

            # criteria of each reader
            mrr_list = []
            map_list = []
            for idx, test_reader_id in enumerate(test['User_id'].tolist()):
                # get the df for test reader, and convert to list
                test_df = test_reader_df[test_reader_df['User_id'] == test_reader_id]
                # test_df = test_df.drop(['User_id'], axis=1).to_numpy().tolist()
                # find the cluster
                # knn_result = model.kneighbors(test_df, return_distance=True)
                cluster = model.predict(test_df.drop(['User_id'], axis=1).to_numpy().tolist())[0]

                # find the similar users in the same cluster
                similar_users_idx = [i for i, x in enumerate(model.labels_) if x == cluster and fold_reader_id_list[i] != test_reader_id]

                # get the similarity between the target user and similar users
                similar_users = [(fold_reader_id_list[i], self.user_similarity(test_df, train_reader_df.iloc[[i]])) for i in similar_users_idx]

                # get n users with highest similarity
                similar_users = sorted(similar_users, key = lambda x: x[1], reverse = True)[:self.user_num]

                # get the match reader id and similarity
                # similar_users = [(fold_reader_id_list[knn_result[1][0][idx]], knn_result[0][0][idx]) for idx in range(self.cluster_num)]

                # get the recommended books that similar users read
                recommended_books = self.recommend_books(similar_users, train_bought_list, test_bought_list[test_reader_id][0])
                # calculate the mrr and map score for the target user
                mrr_q = self.mrr_criteria(test_buying_list[test_reader_id][0], recommended_books)
                map_aveq, hit_location = self.map_criteria(test_buying_list[test_reader_id][0], recommended_books)

                mrr_list.append(mrr_q)
                map_list.append(map_aveq)
                print("{} th Target User: ".format(idx), test_reader_id)
                if not __debug__:
                    print("Target Books: ", test_buying_list[test_reader_id][0])
                    print("Recommended Books: ", recommended_books)
                print("MRR: ", mrr_q)
                print("MAP: ", map_aveq)
                print("Hit Location: ", hit_location)
                print("=========================================")
            print("[CV{}/{}] MRR Score: ".format(fold_round + 1, n), np.mean(mrr_list))
            print("[CV{}/{}] MAP Score: ".format(fold_round + 1, n), np.mean(map_list))

            mrr_cv.append(np.mean(mrr_list))
            map_cv.append(np.mean(map_list))

        print("=========================================")
        print("each fold MRR Score: ", mrr_cv)
        print("each fold MAP Score: ", map_cv)
        print("=========================================")
        print("Final MRR Score: ", np.mean(mrr_cv))
        print("Final MAP Score: ", np.mean(map_cv))
        print("=========================================")
        print("Best MRR Score: ", np.max(mrr_cv))
        print("Best MAP Score: ", np.max(map_cv))

    def process(self):
        self.data = self.load_data()
        # self.plot_pca()
        self.kMeans()