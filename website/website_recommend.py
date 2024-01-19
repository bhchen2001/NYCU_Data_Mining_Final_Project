from website_knn import bookKNN
from website_user_based_cf import UserBasedCF
from item_base import itemBasedCF
from optparse import OptionParser
import json
import threading
import pandas as pd
import re

class websiteRecommend():
    def __init__(self, book_list):
        self.data_path = "./dataset/"
        self.book_list = book_list
        self.item_cf_result = None
        self.user_cf_result = None
        self.knn_result = None

    def knn(self):
        print("=== KNN ===")
        my_knn = bookKNN(self.data_path, self.book_list, k = 20, recommend_num = 30)
        self.knn_result = my_knn.process()

    def item_based_cf(self):
        print("=== Item Based CF ===")
        my_item_based_cf = itemBasedCF(self.data_path, self.book_list, recommend_num = 30)
        self.item_cf_result = my_item_based_cf.process()

    def user_based_cf(self):
        print("=== User Based CF ===")
        my_user_based_cf = UserBasedCF(self.data_path, self.book_list, "cosine_similarity", k = 20, recommend_num = 30)
        self.user_cf_result = my_user_based_cf.process()

    def strip_title(self):
        for idx, title in enumerate(self.knn_result):
            tmp = title.strip('\"\',.!?()[]{}<>:;')
            self.knn_result[idx] = tmp
        for idx, title in enumerate(self.item_cf_result):
            tmp = title.strip('\"\',.!?()[]{}<>:;')
            self.item_cf_result[idx] = tmp
        for idx, title in enumerate(self.user_cf_result):
            tmp = title.strip('\"\',.!?()[]{}<>:;')
            self.user_cf_result[idx] = tmp
        

    def run(self):
        t_knn = threading.Thread(target = self.knn)
        t_item_based_cf = threading.Thread(target = self.item_based_cf)
        t_user_based_cf = threading.Thread(target = self.user_based_cf)

        t_knn.start()
        t_item_based_cf.start()
        t_user_based_cf.start()

        t_knn.join()
        t_item_based_cf.join()
        t_user_based_cf.join()

    def combine(self):

        # # combine the result of knn and user based cf
        # # sort the result by oppear times then by score 
        result_dict = []
        for idx, book in enumerate(self.knn_result):
            tmp_dict = {
                "Title": book,
                "Appear time": 1,
                "Score": 1 / (idx + 1) * 0.25
            }
            result_dict.append(tmp_dict)

        for idx, book in enumerate(self.user_cf_result):
            # if book exist in result_dict['Title'], then add the appear time and score
            # else add the book to result_dict
            if book in [book_dict['Title'] for book_dict in result_dict]:
                result_dict[[book_dict['Title'] for book_dict in result_dict].index(book)]["Appear time"] += 1
                result_dict[[book_dict['Title'] for book_dict in result_dict].index(book)]["Score"] += 1 / (idx + 1)
            else:
                tmp_dict = {
                    "Title": book,
                    "Appear time": 1,
                    "Score": 1 / (idx + 1) * 0.25
                }
                result_dict.append(tmp_dict)

        for idx, book in enumerate(self.item_cf_result):
            # if book exist in result_dict['Title'], then add the appear time and score
            # else add the book to result_dict
            if book in [book_dict['Title'] for book_dict in result_dict]:
                result_dict[[book_dict['Title'] for book_dict in result_dict].index(book)]["Appear time"] += 1
                result_dict[[book_dict['Title'] for book_dict in result_dict].index(book)]["Score"] += 1 / (idx + 1)
            else:
                tmp_dict = {
                    "Title": book,
                    "Appear time": 1,
                    "Score": 1 / (idx + 1) * 0.5
                }
                result_dict.append(tmp_dict)
        for book in result_dict:
            book["Avg_Score"] = book['Score'] / book["Appear time"] if book["Appear time"] != 0 else 0
        result_dict = sorted(result_dict, key = lambda x: (x["Appear time"], x["Avg_Score"]), reverse = True)
        
        # print(result_dict)

        self.hybrid_result = [book['Title'] for book in result_dict][:10]
    
    def book_info(self):
        book_df = pd.read_csv(self.data_path + 'books_info_full.csv')
        # get row with title in book_list
        book_df = book_df[book_df['Title'].isin(self.hybrid_result)]

        recommend_dict = []
        for book in self.hybrid_result:
            tmp_df = book_df[book_df['Title'] == book].copy()
            tmp_df.fillna("Unknown", inplace = True)
            tmp_dict = {
                "Title": book,
                "Author": tmp_df['authors'].tolist()[0],
                "Categories": tmp_df['categories'].tolist()[0],
                "Publisher": tmp_df['publisher'].tolist()[0],
                "PublishedDate": tmp_df['publishedDate'].tolist()[0],
                "Score": float("{:.2f}".format(float(tmp_df['review/score'].tolist()[0])))
            }
            recommend_dict.append(tmp_dict)

        print(recommend_dict)

        # turn recommend_dict into json
        json_data = json.dumps(recommend_dict)
        # write file
        with open('recommend.json', 'w') as f:
            f.write(json_data)
        return json_data
    
    def process(self):
        self.run()
        # self.strip_title()
        self.combine()
        return self.book_info()

if __name__ == '__main__':

    book_list = ['dr. seuss: american icon', "rising sons and daughters: life among japan's new young"]

    my_website_recommend = websiteRecommend(book_list)
    json_data = my_website_recommend.process()

    print(json_data)