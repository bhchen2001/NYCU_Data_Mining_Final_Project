import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class bookKNN():
    def __init__(self, data_path, k = 5, r = 3):
        self.data_path = data_path
        self.data = None
        self.reader_list = None
        self.category_list = None
        self.reader_df = None
        # store the buying list and the review score of each reader
        # sorted by review score
        self.reader_buying_list = {}
        self.k_nearest = k
        self.r_recommend = r

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data
    
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

    def preprocessing(self):
        # normalize the numerical data (min/max normalization)
        self.data['Price'] = (self.data['Price'] - self.data['Price'].min()) / (self.data['Price'].max() - self.data['Price'].min())
        # self.data['review/score'] = (self.data['review/score'] - self.data['review/score'].min()) / (self.data['review/score'].max() - self.data['review/score'].min())
        self.data['ratingsCount'] = (self.data['ratingsCount'] - self.data['ratingsCount'].min()) / (self.data['ratingsCount'].max() - self.data['ratingsCount'].min())

        # get the year from the date
        self.data['publishedDate'] = self.data['publishedDate'].str.split('-').str[0]
        # fill the missing value with the most frequent year
        self.data['publishedDate'] = self.data['publishedDate'].fillna(self.data['publishedDate'].value_counts().index[0])
        # handle the year with ?, replace with the most frequent year of that century (19xx)
        self.data.loc[self.data['publishedDate'].str.find("?") != -1, 'publishedDate'] = self.data.loc[self.data['publishedDate'].str[:2] == '19', 'publishedDate'].value_counts().index[0]
        # print(self.data.loc[self.data['publishedDate'].str.find("?") != -1, 'publishedDate'])
        # print(self.data.loc[self.data['publishedDate'].str[:2] == '19', 'publishedDate'].value_counts().index[0])
        self.data['publishedDate'] = self.data['publishedDate'].astype(int)
        
        # one-hot encoding for book category
        self.data['categories'] = self.data['categories'].str.replace('[', '').str.replace(']', '').str.replace("'", '')
        self.category_list = self.data['categories'].unique()
        self.data = pd.get_dummies(self.data, columns=['categories'])

        # helpfulness ratio
        self.data['review/helpfulness'] = self.data['review/helpfulness'].str.split('/').str[0].astype(float) / self.data['review/helpfulness'].str.split('/').str[1].astype(float)
        self.data['review/helpfulness'] = self.data['review/helpfulness'].fillna(0)

        # build the reader list with User_id
        self.reader_list = self.data['User_id'].unique()

    def build_reader_df(self):
        # build dataframe and buying list for each reader
        attribute_dict = {
            'User_id': [],
            'avg_helpfulness': [],
            'start_buying_year': [],
            'end_buying_year': [],
        }
        for category in self.category_list:
            attribute_dict["num_" + category] = []
            attribute_dict["score_" + category] = []
        for reader in self.reader_list:
            attribute_dict['User_id'].append(reader)
            self.reader_buying_list[reader] = [[], []]
            tmp_df = self.data[self.data['User_id'] == reader]
            attribute_dict['avg_helpfulness'].append(tmp_df['review/helpfulness'].mean())
            attribute_dict['start_buying_year'].append(tmp_df['publishedDate'].min())
            attribute_dict['end_buying_year'].append(tmp_df['publishedDate'].max())
            for category in self.category_list:
                attribute_dict["num_" + category].append(tmp_df['categories_' + category].sum())
                score_mean = tmp_df[tmp_df['categories_' + category] == 1]['review/score'].mean()
                attribute_dict["score_" + category].append(0 if np.isnan(score_mean) else score_mean)
            # buying list for each reader sorted by review score
            self.reader_buying_list[reader][0] = tmp_df.sort_values(by=['review/score'], ascending=False)['Id'].tolist()
            self.reader_buying_list[reader][1] = tmp_df.sort_values(by=['review/score'], ascending=False)['review/score'].tolist()

        # for i in range(0, 1):
        #     print("reader id: ", self.reader_list[i])
        #     print("buying list: ", self.reader_buying_list[self.reader_list[i]][0])
        #     print("buying review: ", self.reader_buying_list[self.reader_list[i]][1])

        reader_df = pd.DataFrame(attribute_dict)

        # normalize the numerical data (min/max normalization)
        reader_df['start_buying_year'] = (reader_df['start_buying_year'] - reader_df['start_buying_year'].min()) / (reader_df['start_buying_year'].max() - reader_df['start_buying_year'].min())
        reader_df['end_buying_year'] = (reader_df['end_buying_year'] - reader_df['end_buying_year'].min()) / (reader_df['end_buying_year'].max() - reader_df['end_buying_year'].min())
        for category in self.category_list:
            reader_df["num_" + category] = (reader_df["num_" + category] - reader_df["num_" + category].min()) / (reader_df["num_" + category].max() - reader_df["num_" + category].min())
            reader_df["score_" + category] = (reader_df["score_" + category] - reader_df["score_" + category].min()) / (reader_df["score_" + category].max() - reader_df["score_" + category].min())
        return reader_df
    
    def map_aveq_cal(self, test_book_list, recommend_book_list):
        # calculate the average precision
        hit = 0
        map_aveq = 0
        for idx, book in enumerate(recommend_book_list):
            if book in test_book_list:
                hit += 1
                map_aveq += hit / (idx + 1)
        if hit != 0:
            map_aveq /= hit
        return map_aveq
    
    def criteria(self, test_user, match_user):
        # book id as key, value contains the frequency, overall score and average score
        recommend_list = {}
        # find the match user's buying list
        for idx_user, user in enumerate(match_user):
            for idx_book, book in enumerate(self.reader_buying_list[user][0]):
                if book not in recommend_list:
                    recommend_list[book] = [1, self.reader_buying_list[user][1][idx_book] * 1 - (idx_user / len(match_user)), 0]
                else:
                    recommend_list[book][0] += 1
                    recommend_list[book][1] += self.reader_buying_list[user][1][idx_book] * 1 - (idx_user / len(match_user))
        # calculate the average review score
        for book in recommend_list:
            recommend_list[book][2] = recommend_list[book][1] / recommend_list[book][0]
        # first sorted by frequency, then by average score
        recommend_list = sorted(recommend_list.items(), key=lambda x: (x[1][0], x[1][2]), reverse=True)
        test_book_list = self.reader_buying_list[test_user][0]
        recommend_book_list = [recommend_list[i][0] for i in range(0, len(recommend_list))]

        # mrr: find the match index of the test book list in the recommend book list
        mrr_q = 0
        for idx_r, book in enumerate(recommend_book_list):
            if book in test_book_list:
                mrr_q = 1 / (idx_r + 1)
                break
        # map: calculate the average precision
        map_aveq = self.map_aveq_cal(test_book_list, recommend_book_list)

        return mrr_q, map_aveq
    
    def knn(self):
        # split the data into training and testing
        train, test = train_test_split(self.reader_df, test_size=0.2, random_state=42)
        train_x = train.drop(['User_id'], axis=1).to_numpy().tolist()
        train_y = train['User_id'].to_numpy().tolist()
        test_x = test.drop(['User_id'], axis=1).to_numpy().tolist()
        test_y = test['User_id'].to_numpy().tolist()

        # train the model
        model = NearestNeighbors(n_neighbors=self.k_nearest)
        model.fit(train_x)

        # for mrr
        mrr_q = []
        # for map
        map_aveq = []
        for idx in range(0, 1):
            # print(test_x.iloc[idx].values.flatten().tolist())
            knn_result = model.kneighbors([test_x[idx]], return_distance=True)
            # print(knn_result)
            # get the match user id
            match_user = [train_y[i] for i in knn_result[1][0]]
            # calculate the r highest recommend books and the evaluation score
            pq, avepq = self.criteria(test_y[idx], match_user)
            mrr_q.append(pq)
            map_aveq.append(avepq)
        print("MRR: ", np.sum(mrr_q) / self.r_recommend)
        print("MAP: ", np.mean(map_aveq))

    def process(self):
        self.data = self.load_data()
        self.data_details(self.data, "Original Data")
        self.preprocessing()
        self.reader_df = self.build_reader_df()
        self.data_details(self.reader_df, "Reader Data")
        self.knn()

if __name__ == '__main__':
    data_path = './dataset/validate_user.csv'
    my_knn = bookKNN(data_path, k=5)
    my_knn.process()