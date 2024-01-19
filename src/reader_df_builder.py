import pandas as pd
import numpy as np
import json
import math

class ReaderDFBuilder():
    """
    Constructor
    Input:
        data_path: 
            type: string
            description: the path of the data folder
    """
    def __init__(self, data_path):
        self.data_path = data_path
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
    Preprocessing
        1. drop the unnecessary columns
        2. get year from publishedDate
        3. one-hot encoding for book category
        4. calculate helpfulness ratio
    """
    def preprocessing(self):
        # drop the unnecessary columns
        self.data = self.data.drop(['Price', 'profileName', 'review/summary', 'review/text', \
                                                      'description', 'authors', 'publisher'], axis = 1)
        # get year from publishedDate
        self.data['publishedDate'] = self.data['publishedDate'].str.split('-').str[0]
        # fill the missing value with the most frequent year
        self.data['publishedDate'] = self.data['publishedDate'].fillna(self.data['publishedDate'].value_counts().index[0])
        # handle the year with ?, replace with the most frequent year of that century (19xx)
        self.data.loc[self.data['publishedDate'].str.find("?") != -1, 'publishedDate'] = self.data.loc[self.data['publishedDate'].str[:2] == '19', 'publishedDate'].value_counts().index[0]
        # remove the * sign in the publishedDate
        self.data['publishedDate'] = self.data['publishedDate'].str.replace('*', '')
        self.data['publishedDate'] = self.data['publishedDate'].astype(int)

        # one-hot encoding for book category
        self.data['categories'] = self.data['categories'].str.replace('[', '').str.replace(']', '').str.replace("'", '')
        # build category list
        self.category_list = self.data['categories'].unique().tolist()
        self.data = pd.get_dummies(self.data, columns=['categories'])

        # helpfulness ratio
        self.data['review/helpfulness'] = self.data['review/helpfulness'].str.split('/').str[0].astype(float) / self.data['review/helpfulness'].str.split('/').str[1].astype(float)
        self.data['review/helpfulness'] = self.data['review/helpfulness'].fillna(0)

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
    def build_reader_df(self, train_reader_id_list, test_reader_id_list):
        # build dataframe and buying list for each reader in the reader_id_list
        attribute_dict = {
            'User_id': [],
            'avg_helpfulness': [],
            'start_buying_year': [],
            'end_buying_year': [],
        }
        train_reader_buying_list = {}
        test_reader_buying_list = {}
        test_reader_bought_list = {}
        for category in self.category_list:
            attribute_dict["ratio_" + category] = []
            attribute_dict["score_" + category] = []
        for reader in train_reader_id_list + test_reader_id_list:
            attribute_dict['User_id'].append(reader)
            tmp_df = self.data[self.data['User_id'] == reader]
            if reader not in self.data['User_id'].unique():
                # error handling
                print("Reader ID: ", reader, " not in the data")
                return
            elif reader in test_reader_id_list:
                """
                random select 50% of the buying list as the test set
                and the rest 50% as the input for predicting
                """
                test_df = tmp_df.sample(frac=0.5, random_state=42)
                tmp_df = tmp_df.drop(test_df.index)

                """
                sort the book list by review time and select the first 50% as the test set
                and the rest 50% as the input for predicting
                """
                # test_df = tmp_df.sort_values(by=['review/time'], ascending=False).head(int(tmp_df.shape[0] / 2))
                # tmp_df = tmp_df.drop(test_df.index)

                """
                consider both review score and bert score
                """
                # buying list for each reader sorted by bert score * review score
                # test_df['combined_score'] = test_df.apply(lambda x: math.sqrt(x['bert_score'] * x['review/score']), axis=1)
                # tmp_df['combined_score'] = tmp_df.apply(lambda x: math.sqrt(x['bert_score'] * x['review/score']), axis=1)

                # buying list for each reader sorted by bert score *0.5 + review score * 0.5
                test_df['combined_score'] = test_df.apply(lambda x: x['bert_score'] * 0.3 + x['review/score'] * 0.7, axis=1)
                tmp_df['combined_score'] = tmp_df.apply(lambda x: x['bert_score'] * 0.3 + x['review/score'] * 0.7, axis=1)

                test_reader_buying_list[reader] = [[], []]

                # buying list for each reader sorted by review score
                # test_reader_buying_list[reader][0] = test_df.sort_values(by=['review/score'], ascending=False)['Title'].tolist()
                # test_reader_buying_list[reader][1] = test_df.sort_values(by=['review/score'], ascending=False)['review/score'].tolist()

                # test_reader_buying_list[reader][0] = test_df.sort_values(by=['bert_score'], ascending=False)['Title'].tolist()
                # test_reader_buying_list[reader][1] = test_df.sort_values(by=['bert_score'], ascending=False)['bert_score'].tolist()

                test_reader_buying_list[reader][0] = test_df.sort_values(by=['combined_score'], ascending=False)['Title'].tolist()
                test_reader_buying_list[reader][1] = test_df.sort_values(by=['combined_score'], ascending=False)['combined_score'].tolist()
                
                test_reader_bought_list[reader] = [[], []]

                # test_reader_bought_list[reader][0] = tmp_df.sort_values(by=['review/score'], ascending=False)['Title'].tolist()
                # test_reader_bought_list[reader][1] = tmp_df.sort_values(by=['review/score'], ascending=False)['review/score'].tolist()

                # test_reader_bought_list[reader][0] = tmp_df.sort_values(by=['bert_score'], ascending=False)['Title'].tolist()
                # test_reader_bought_list[reader][1] = tmp_df.sort_values(by=['bert_score'], ascending=False)['bert_score'].tolist()

                test_reader_bought_list[reader][0] = tmp_df.sort_values(by=['combined_score'], ascending=False)['Title'].tolist()
                test_reader_bought_list[reader][1] = tmp_df.sort_values(by=['combined_score'], ascending=False)['combined_score'].tolist()
            elif reader in train_reader_id_list:
                """
                consider both review score and bert score
                """
                # tmp_df['combined_score'] = tmp_df.apply(lambda x: math.sqrt(x['bert_score'] * x['review/score']), axis=1)
                tmp_df['combined_score'] = tmp_df.apply(lambda x: x['bert_score'] * 0.3 + x['review/score'] * 0.7, axis=1)

                train_reader_buying_list[reader] = [[], []]
                # buying list for each reader sorted by review score
                # train_reader_buying_list[reader][0] = tmp_df.sort_values(by=['review/score'], ascending=False)['Title'].tolist()
                # train_reader_buying_list[reader][1] = tmp_df.sort_values(by=['review/score'], ascending=False)['review/score'].tolist()

                # train_reader_buying_list[reader][0] = tmp_df.sort_values(by=['bert_score'], ascending=False)['Title'].tolist()
                # train_reader_buying_list[reader][1] = tmp_df.sort_values(by=['bert_score'], ascending=False)['bert_score'].tolist()

                train_reader_buying_list[reader][0] = tmp_df.sort_values(by=['combined_score'], ascending=False)['Title'].tolist()
                train_reader_buying_list[reader][1] = tmp_df.sort_values(by=['combined_score'], ascending=False)['combined_score'].tolist()
            attribute_dict['avg_helpfulness'].append(tmp_df['review/helpfulness'].mean())
            attribute_dict['start_buying_year'].append(tmp_df['publishedDate'].min())
            attribute_dict['end_buying_year'].append(tmp_df['publishedDate'].max())
            for category in self.category_list:
                # ratio of buying books in each category
                attribute_dict["ratio_" + category].append(tmp_df['categories_' + category].sum() / tmp_df.shape[0])
                # average review score of buying books in each category
                score_mean = tmp_df[tmp_df['categories_' + category] == 1]['review/score'].mean()
                attribute_dict["score_" + category].append(0 if np.isnan(score_mean) else score_mean)

        reader_df = pd.DataFrame(attribute_dict)
        # normalize the data
        reader_df['avg_helpfulness'] = (reader_df['avg_helpfulness'] - reader_df['avg_helpfulness'].min()) / (reader_df['avg_helpfulness'].max() - reader_df['avg_helpfulness'].min())
        reader_df['start_buying_year'] = (reader_df['start_buying_year'] - reader_df['start_buying_year'].min()) / (reader_df['start_buying_year'].max() - reader_df['start_buying_year'].min())
        reader_df['end_buying_year'] = (reader_df['end_buying_year'] - reader_df['end_buying_year'].min()) / (reader_df['end_buying_year'].max() - reader_df['end_buying_year'].min())
        for category in self.category_list:
            reader_df["ratio_" + category] = (reader_df["ratio_" + category] - reader_df["ratio_" + category].min()) / (reader_df["ratio_" + category].max() - reader_df["ratio_" + category].min())
            reader_df["score_" + category] = (reader_df["score_" + category] - reader_df["score_" + category].min()) / (reader_df["score_" + category].max() - reader_df["score_" + category].min())

        # split the data into training and testing set
        train_reader_df = reader_df[reader_df['User_id'].isin(train_reader_id_list)]
        test_reader_df = reader_df[reader_df['User_id'].isin(test_reader_id_list)]
        
        # reset the index of the dataframe
        train_reader_df = train_reader_df.reset_index(drop=True)
        test_reader_df = test_reader_df.reset_index(drop=True)

        return train_reader_df, test_reader_df, train_reader_buying_list, test_reader_buying_list, test_reader_bought_list

    def process(self):
        self.load_data()
        self.data_details(self.data, "Original Data")
        self.preprocessing()