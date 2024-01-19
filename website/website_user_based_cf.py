import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from website_reader_df_builder import ReaderDFBuilder

class UserBasedCF():
    """
    Constructor
    Input:
        data_path: 
            type: string
            description: the path of the data folder
        similarity_method:
            type: string
            description: the similarity method used for finding similar users
            example: "cosine_similarity"
        k:
            type: int
            description: the number of nearest neighbors
        recommend_num:
            type: int
            description: the number of recommended books
    """
    def __init__(self, data_path, book_list, similarity_method, k = 3, recommend_num = 10):
        self.similarity_method = similarity_method
        self.k_nearest = int(k)
        self.data_path = data_path
        self.recommend_num = recommend_num
        self.book_list = book_list

    """
    Read reader id file
    """
    def load_data(self):
        self.reader_list = pd.read_csv(self.data_path + 'validate_reader_id.csv')

    """
    Find Similar Users Depend on the Similarity Method
    Input:
        test_reader_df:
            type: pandas dataframe
            description: the dataframe of the test reader
        train_reader_df:
            type: pandas dataframe
            description: the dataframe of the train readers

    Output:
        similarity_list[:k_nearest]:
            type: list
            description: the list of k most similar users
            example: [(user_id, similarity), ...]
    """
    def find_similar_users(self, test_reader_df, train_reader_df):
        # drop the User_id feature
        test_reader_df = test_reader_df.drop(['User_id'], axis = 1)
        # find the similarity between test reader and all other train readers
        similarity_list = []
        if self.similarity_method == "cosine_similarity":
            for index, train_df in train_reader_df.iterrows():
                # drop the User_id column
                user_id = train_df['User_id']
                train_df = train_df.drop(['User_id'])
                similarity = cosine_similarity(test_reader_df.values.reshape(1, -1), train_df.values.reshape(1, -1))
                similarity_list.append((user_id, similarity[0][0]))
        # sort the list by similarity
        similarity_list.sort(key = lambda x: x[1], reverse = True)

        # return the top k similar readers
        return similarity_list[:self.k_nearest]
    
    """
    Recommend Books Depend on the Similar Users
        the recommended books are sorted by the occurence and average score
    Input:
        similar_users:
            type: list
            description: the list of similar users
            example: [(user_id, similarity), ...]
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
        for user in similar_users:
            user_id, user_similarity = user[0], user[1]
            user_buying_list = train_bought_list[user_id]
            for book_idx, book in enumerate(user_buying_list[0]):
                if book in test_bought_list:
                    # the test user has already bought this book
                    continue
                if book not in recommend_book_dict:
                    recommend_book_dict[book] = {
                        'occurence': 1,
                        # the score also consider the similarity between users
                        'score': user_buying_list[1][book_idx] * user_similarity,
                        'avg_score': 0
                    }
                else:
                    recommend_book_dict[book]['occurence'] += 1
                    recommend_book_dict[book]['score'] += user_buying_list[1][book_idx] * user_similarity
        # calculate the average score of each book
        for book_id, book in recommend_book_dict.items():
            book['avg_score'] = book['score'] / book['occurence']
        # sort the books first by occurence, then by average score
        recommend_book_list = sorted(recommend_book_dict.items(), key = lambda x: (x[1]['occurence'], x[1]['avg_score']), reverse = True)
        # return the top r books id
        return [book[0] for book in recommend_book_list[:self.recommend_num]]
    
    def recommend_books_secret(self, similar_users, train_bought_list, test_bought_list):
        # for each similar user, create a list of books that he/she read
        # for each list, sort the books by review score
        recommend_book_list = []
        for user in similar_users:
            user_id, user_similarity = user[0], user[1]
            user_buying_list = train_bought_list[user_id]
            # sort the books by review score
            sorted_books = sorted(zip(user_buying_list[0], user_buying_list[1]), key = lambda x: x[1], reverse = True)
            # get the book id list
            sorted_books = [book[0] for book in sorted_books]
            # get the top 25 books
            sorted_books = sorted_books[:10]
            recommend_book_list.append(sorted_books)
        return recommend_book_list
    
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
    
    def mrr_criteria_secret(self, target_book_list, recommend_books):
        mrr_q = 0
        for book_list in recommend_books:
            for idx, book in enumerate(book_list):
                if book in target_book_list:
                    mrr_q = 1 / (idx + 1)
                    return mrr_q
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
    User Based Collaborative Filtering
        1. split the data into training and testing set
        2. build reader dataframe and buying list
        3. for each user in the testing set, find the top k similar users in the training set
            and find the books that similar users read
            then recommend the books to the target user
        4. calculate the mrr and map score for the target user
        5. output the average mrr and map score
    """
    def user_based_cf(self):

        train = self.reader_list
        test_id = 'OCSHCEANR'
    
        # split the data into training and testing set
        # train, test = train_test_split(self.reader_list, test_size = 0.2, random_state = 42)

        # build reader dataframe and buying list
        reader_df_builder = ReaderDFBuilder(self.data_path, self.book_list, test_id)
        reader_df_builder.process()
        train_reader_df, test_reader_df, train_bought_list, test_bought_list = \
            reader_df_builder.build_reader_df()

        # for each user in the testing set, find the top k similar users in the training set
        # and find the books that similar users read
        # then recommend the books to the target user
        similar_users = self.find_similar_users(test_reader_df, train_reader_df)
        # get the recommended books that similar users read
        recommended_books = self.recommend_books(similar_users, train_bought_list, test_bought_list[test_id][0])

        print(recommended_books)

        return recommended_books

    def process(self):
        self.load_data()
        return self.user_based_cf()