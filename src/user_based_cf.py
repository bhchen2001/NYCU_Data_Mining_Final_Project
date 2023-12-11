import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from reader_df_builder import ReaderDFBuilder

class UserBasedCF():
    def __init__(self, data_path, similarity_method, k = 3, recommand_num = 20):
        self.similarity_method = similarity_method
        self.k_nearest = int(k)
        self.data_path = data_path
        self.recommand_num = recommand_num

    def load_data(self):
        self.reader_list = pd.read_csv(self.data_path + 'validate_reader_id.csv')

    def find_similar_users(self, test_reader_df, train_reader_df):
        # drop the User_id column
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
        # print(similarity_list)

        # return the top k similar readers
        return similarity_list[:self.k_nearest]
    
    def recommend_books(self, similar_users, train_reader_list):
        # for each similar user, find the books that he/she read
        # and calculate the occurence, total score and average score of each book
        recommend_book_dict = {}
        for user in similar_users:
            user_id, user_similarity = user[0], user[1]
            user_buying_list = train_reader_list[user_id]
            for book_idx, book in enumerate(user_buying_list[0]):
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
        # print(recommend_book_list)
        # return the top 10 books id
        return [book[0] for book in recommend_book_list[:self.recommand_num]]
    
    def mrr_criteria(self, target_book_list, recommended_books):
        mrr_q = 0
        for idx, book in enumerate(recommended_books):
            if book in target_book_list:
                mrr_q = 1 / (idx + 1)
                break
        return mrr_q
    
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

    def user_based_cf(self):
        # split the data into training and testing set
        train, test = train_test_split(self.reader_list, test_size = 0.2, random_state = 42)

        # build reader dataframe and buying list
        reader_df_builder = ReaderDFBuilder(self.data_path)
        reader_df_builder.process()
        train_reader_df, test_reader_df, train_reader_list, test_reader_list = \
            reader_df_builder.build_reader_df(train['User_id'].tolist(), test['User_id'].tolist())
        
        # print the length of training and testing set
        print("Training Set Length: ", len(train_reader_list))
        print("Testing Set Length: ", len(test_reader_list))
        print("Training buying list length: ", len(train_reader_df))
        print("Testing buying list length: ", len(test_reader_df))

        # for each user in the testing set, find the top k similar users in the training set
        # and find the books that similar users read
        # then recommand the books to the target user
        mrr_list = []
        map_list = []
        for idx, test_reader_id in enumerate(test['User_id'].tolist()):
            similar_users = self.find_similar_users(test_reader_df[test_reader_df['User_id'] == test_reader_id], train_reader_df)
            # get the recommended books that similar users read
            recommended_books = self.recommend_books(similar_users, train_reader_list)
            # calculate the mrr and map score for the target user
            mrr_q = self.mrr_criteria(test_reader_list[test_reader_id][0], recommended_books)
            map_aveq = self.map_criteria(test_reader_list[test_reader_id][0], recommended_books)

            mrr_list.append(mrr_q)
            map_list.append(map_aveq)
            print("{} th Target User: ".format(idx), test_reader_id)
            # print("Target Books: ", test_reader_list[test_reader_id][0])
            # print("Recommended Books: ", recommended_books)
            print("MRR: ", mrr_q)
            print("MAP: ", map_aveq)
            print("=========================================")
        print("MRR Score: ", np.mean(mrr_list))
        print("MAP Score: ", np.mean(map_list))

    def process(self):
        self.load_data()
        self.user_based_cf()