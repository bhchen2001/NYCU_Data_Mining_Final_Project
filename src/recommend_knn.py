import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import json

from reader_df_builder import ReaderDFBuilder

class bookKNN():
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
    def __init__(self, data_path, k = 3, recommend_num = 10):
        self.data_path = data_path
        self.reader_list = None
        self.k_nearest = k
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
        map_aveq = 0
        for idx, book in enumerate(recommended_books):
            if book in target_book_list:
                hit += 1
                map_aveq += hit / (idx + 1)
        if hit != 0:
            map_aveq /= hit
        return map_aveq

    """
    KNN Algorithm
        1. split the data into training and testing
        2. build reader dataframe and buying list
        3. fit the KNN model
        4. for each user in the testing set, find the top k similar users in the training set
            and find the books that similar users read
        5. calculate the mrr and map score for the target user
        6. output the average mrr and map score
    """
    def knn(self):
        # split the data into training and testing
        train, test = train_test_split(self.reader_list, test_size=0.2, random_state=42)
        fold_reader_id_list = train['User_id'].tolist()

        # build reader dataframe and buying list
        reader_df_builder = ReaderDFBuilder(self.data_path)
        reader_df_builder.process()
        train_reader_df, test_reader_df, train_bought_list, test_buying_list, test_bought_list = \
            reader_df_builder.build_reader_df(train['User_id'].tolist(), test['User_id'].tolist())
        
        # train the model
        model = NearestNeighbors(n_neighbors=self.k_nearest)
        model.fit(train_reader_df.drop(['User_id'], axis=1).to_numpy().tolist())

        # criteria of each reader
        mrr_list = []
        map_list = []
        for idx, test_reader_id in enumerate(test['User_id'].tolist()):
            # get the df for test reader, and convert to list
            test_df = test_reader_df[test_reader_df['User_id'] == test_reader_id]
            test_df = test_df.drop(['User_id'], axis=1).to_numpy().tolist()
            # find the k nearest neighbors
            knn_result = model.kneighbors(test_df, return_distance=True)
            # get the match reader id and similarity
            similar_users = [(fold_reader_id_list[knn_result[1][0][idx]], knn_result[0][0][idx]) for idx in range(self.k_nearest)]

            # get the recommended books that similar users read
            recommended_books = self.recommend_books(similar_users, train_bought_list, test_bought_list[test_reader_id][0])
            # calculate the mrr and map score for the target user
            mrr_q = self.mrr_criteria(test_buying_list[test_reader_id][0], recommended_books)
            map_aveq = self.map_criteria(test_buying_list[test_reader_id][0], recommended_books)

            mrr_list.append(mrr_q)
            map_list.append(map_aveq)
            print("{} th Target User: ".format(idx), test_reader_id)
            if not __debug__:
                print("Target Books: ", test_buying_list[test_reader_id][0])
                print("Recommended Books: ", recommended_books)
            print("MRR: ", mrr_q)
            print("MAP: ", map_aveq)
            print("=========================================")
        print("MRR Score: ", np.mean(mrr_list))
        print("MAP Score: ", np.mean(map_list))

    def process(self):
        self.data = self.load_data()
        self.knn()