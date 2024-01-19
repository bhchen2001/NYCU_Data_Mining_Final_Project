import pandas as pd
import numpy as np
import json
import math
import pickle

class ReaderDFBuilder():
    """
    Constructor
    Input:
        data_path: 
            type: string
            description: the path of the data folder
    """
    def __init__(self, data_path, book_list, test_id):
        self.data_path = data_path
        self.book_list = book_list
        self.test_id = test_id
        self.data = None

    """
    Read the data
    """
    def load_data(self):
        self.data = pd.read_csv(self.data_path + 'validate_data_with_bert.csv')

    """
    Show the Processed Data Details
    Input:
        df:
            type: pandas dataframe
            description: the dataframe of the data
        df_name:
            type: string
            description: the name of the dataframe
    """
    def data_details(self, df, df_name):
        print("Data Details of: ", df_name)
        print("----------------------------------------")
        print(df.describe())
        print(df.info())
        print("----------------------------------------")
        print("Missing Values: ")
        print(df.isnull().sum())
        print("----------------------------------------")
        print("Data Head: ")
        print(df.head(3))

    """
    Build Reader Dataframe and Buying List
    Input:
        train_reader_id_list:
            type: list
            description: the list of reader id in the training set
            example: [reader_id, ...]
        test_reader_id_list:
            type: list
            description: the list of reader id in the testing set
            example: [reader_id, ...]
    Output:
        train_reader_df:
            type: pandas dataframe
            description: the dataframe of the training set
        test_reader_df:
            type: pandas dataframe
            description: the dataframe of the testing set
        train_reader_buying_list:
            type: dictionary
            description: the buying list of each reader in the training set
            example: {reader_id: [[book_id, ...], [review_score, ...]], ...}
        test_reader_buying_list:
            type: dictionary
            description: part of buying list of each reader in the testing set, used for predicting
            example: {reader_id: [[book_id, ...], [review_score, ...]], ...}
        test_reader_bought_list:
            type: dictionary
            description: part of buying list of each reader in the testing set, 
                         used for building the user profile
            example: {reader_id: [[book_id, ...], [review_score, ...]], ...}
    """
    def build_reader_df(self):

        train_reader_df = pd.read_csv(self.data_path + 'reader_info.csv')
        train_reader_buying_list = pickle.load(open(self.data_path + 'reader_buying_list.pkl', 'rb'))
        
        test_reader_bought_list = {self.test_id: [self.book_list, [5 for i in range(len(self.book_list))]]}
        book_info = pd.read_csv(self.data_path + 'book_info.csv')
        self.category_list = book_info['categories'].unique().tolist()
        attribute_dict = {
            'User_id': self.test_id,
            'avg_helpfulness': 0,
            'start_buying_year': None,
            'end_buying_year': None,
        }
        for category in self.category_list:
            attribute_dict["ratio_" + category] = []
            attribute_dict["score_" + category] = []
        tmp_df = book_info[book_info['Title'].isin(self.book_list)]
        # print(tmp_df)
        max_publish_year, min_publish_year = book_info['publishedDate'].max(), book_info['publishedDate'].min()
        attribute_dict['start_buying_year'] = tmp_df['publishedDate'].min()
        attribute_dict['end_buying_year'] = tmp_df['publishedDate'].max()
        category_list = tmp_df['categories'].tolist()
        for category in self.category_list:
            # appearance ratio of buying books in each category
            attribute_dict["ratio_" + category].append(category_list.count(category) / len(category_list))
            # average review score of buying books in each category
            attribute_dict["score_" + category].append(1 if category_list.count(category) > 0 else 0)

        test_reader_df = pd.DataFrame(attribute_dict)
        # print(test_reader_df)
        # normalize the data
        test_reader_df['start_buying_year'] = (test_reader_df['start_buying_year'] - min_publish_year) / (max_publish_year - min_publish_year)
        test_reader_df['end_buying_year'] = (test_reader_df['end_buying_year'] - min_publish_year) / (max_publish_year - min_publish_year)

        # print(test_reader_df)

        return train_reader_df, test_reader_df, train_reader_buying_list, test_reader_bought_list

    def process(self):
        self.load_data()
        # self.data_details(self.data, "Original Data")
        # self.preprocessing()