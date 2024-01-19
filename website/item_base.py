import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import math

class itemBasedCF():
    def __init__(self, data_path, booktitle_list, recommend_num, k = 10):
        self.data_path = data_path
        self.k = k
        self.r = recommend_num
        self.booktitle_list = booktitle_list

    def process(self):
        df2 = pd.read_csv(self.data_path + 'validate_data_with_bert.csv')
        drop_df = pd.read_csv(self.data_path + 'drop_df.csv')

        drop_df_final = drop_df_final = drop_df.drop(['categories'], axis =1)

        # 創建 MinMaxScaler
        scaler = MinMaxScaler()
        # 使用 fit_transform 進行縮放
        tmp = drop_df_final.drop(['Title'], axis = 1)
        scaled_data2 = scaler.fit_transform(tmp)
        # 將縮放後的數據轉換回 DataFrame
        scaled_df2 = pd.DataFrame(scaled_data2, columns=tmp.columns)
        unique_user = df2['User_id'].unique()
        user_with_book = defaultdict(list)
        # Iterate through the DataFrame rows
        for index, row in df2.iterrows():
            user_with_book[row['User_id']].append({
                'review_time': row['review/time'],
                'title': row['Title'],
                'review/score': row['review/score'],
                'bert_score': row['bert_score']
            })
        user_score_sorted = defaultdict(list)
        for x in unique_user:
            for i in range(len(user_with_book[x])):
                user_score_sorted[x].append(math.sqrt(user_with_book[x][i]['review/score'] * user_with_book[x][i]['bert_score']))
        #### ===============user book的title =========================
        # self.booktitle_list = ['love not a rebel', 'maidensong', 'the falcon and the flower', 'the book of the seven delights (jove historical romance)', 'once and always', 'damsel in distress', 'saving grace', 'the highwayman (irish eyes)', 'knight triumphant', 'princess: (book 2 in the ascension trilogy)', 'drums of autumn hardcover first edition', "a pirate's pleasure", 'fair is the rose', 'a hunger like no other (immortals after dark, book 1)', 'bequest', 'behind closed doors', 'earth song', 'sword of the heart', 'petals on the river', 'lord of the wolves', "skye o'malley", 'the border hostage', 'golden surrender', 'the black swan (irish eyes)', 'the pirate lord (the lord trilogy)', 'the dangerous lord (the lord trilogy)', 'straight for the heart', 'on a highland shore', 'whitney, my love', 'then came you', 'the captain of all pleasures', 'sea fire', 'the irish rogue', 'entwined', 'someone to watch over me : a novel', 'defy not the heart (bced)', 'ravished', 'the present (malory family)', 'the bride', 'surrender my love', 'viking passion', "warrior's song (medieval song quartet #4)", 'the darkest heart', 'blue heaven, black night', 'captive of my desires', 'after innocence', 'tender rebel', 'almost heaven', 'no gentle love', 'this is all i ask', 'wild hearts', 'conquer the night (a zebra historical romance)', 'kilgannon', 'the wild rose of kilgannon', 'daring the devil', 'the irish princess (zebra historical romance)', 'embrace and conquer', 'charming the shrew (berkley sensation)', 'believe in me', "passion's ransom", 'pirate of her own, a', 'master of seduction', 'beyond the highland mist', 'to tame a highland warrior (the highlander series, book 2)', 'without honor', 'hawk and the dove', 'heartstorm']
        title_to_index = {title: index for index, title in enumerate(drop_df_final['Title'])}
        user_book_idxs = []
        user_score_idxs = user_score_sorted[unique_user[-1]]
            
        user_book_idxs = [title_to_index.get(booktitle, -1) for booktitle in self.booktitle_list]

        # print(user_book_idxs)
        # 計算餘弦相似度
        similarities = cosine_similarity(scaled_df2)
        # 將相似度矩陣轉換為 DataFrame
        similarities_df = pd.DataFrame(similarities, index=scaled_df2.index, columns=scaled_df2.index)
        def recommend_books_indices(book_index, similarities_df, n=10):
            # 獲取與目標書籍相似度最高的書籍
            similar_books = similarities_df.iloc[book_index].drop(labels=[scaled_df2.index[book_index]])
            top_n_indices = similar_books.nlargest(n).index
            return top_n_indices, similar_books.loc[top_n_indices].tolist()

        # 為一組書籍生成推薦列表
        rec_list = []
        rec_dict = defaultdict(int)
        tmp_set = set(user_book_idxs[:len(user_book_idxs) // 2])
        recb_list = []
        simbval_list = []

        # 假設 user_book_idxs 包含用戶感興趣的書籍索引
        for book_index in tmp_set:    
                recommended_books_indices, similar_books_val = recommend_books_indices(book_index, similarities_df)
                recb_list.append(recommended_books_indices)
                simbval_list.append(similar_books_val)
                for idx in recommended_books_indices:
                    rec_dict[idx] += 1

        for rec_idx, sim_val in zip(recb_list, simbval_list):
            for idx, val in zip(rec_idx, sim_val):
                if idx not in tmp_set:
                    tmp_set.add(idx)
                    rec_list.append({'appear_time': rec_dict[idx], 'val': val, 'idx': idx})

        rec_list = sorted(rec_list, key=lambda x: (x['appear_time'],x['val']), reverse=True)
        rec_list_check = []
        recommended_books_indices_list_final = []
        all_books = drop_df_final['Title'].tolist()
        for i in range(len(rec_list)):
            recommended_books_indices_list_final.append(all_books[rec_list[i]['idx']])
            rec_list_check.append(rec_list[i]['idx'])
        # print(recommended_books_indices_list_final)

        return recommended_books_indices_list_final[:self.r]
        # print(len(rec_list_check))

        # exceptresult = [151, 166, 265, 291, 335, 541, 763, 923, 1127, 1323, 5323, 7648, 18455, 20664, 15069, 15965, 12160, 14894, 22231, 267, 634, 815, 994, 996, 1878, 1895, 1998, 2102, 2315, 200, 631, 1430, 3560, 3653, 4160, 4686, 5115, 6209, 6220, 1336, 2318, 2536, 3210, 3242, 4700, 5830, 6338, 7154, 7273, 1314, 2086, 3525, 3972, 4398, 4460, 5065, 9638, 10057, 10296, 21829, 22882, 53, 1654, 2250, 2509, 2865, 5773, 5779, 6285, 6570, 6722, 12798, 14027, 18206, 25179, 7927, 13544, 24360, 25848, 8439, 12005, 15734, 16156, 19885, 150, 2170, 2708, 3517, 6798, 11318, 13607, 16394, 19985, 21668, 358, 635, 725, 1800, 1806, 2249, 2832, 2946, 3783, 4011, 15019, 16949, 17230, 22108, 1719, 2145, 9224, 11717, 15493, 22112, 91, 1840, 1877, 2048, 2669, 3378, 4603, 5435, 5610, 5832, 1104, 3711, 14349, 15515, 17747, 18769, 20274, 20948, 21524, 23615, 346, 421, 486, 491, 519, 995, 1166, 1291, 1356, 1382, 3707, 6805, 13030, 14653, 20128, 606, 8566, 21231, 353, 732, 850, 1263, 1278, 1474, 1952, 2116, 2298, 2647, 632, 1489, 2689, 2970, 3043, 3633, 3809, 3938, 4110, 4252, 8602, 11280, 12173, 580, 1460, 1707, 1744, 2461, 2490, 3156, 5611, 6183, 7148, 2752, 5578, 11766, 24749, 1047, 3335, 4829, 13822, 5351, 9611, 13017, 2572, 3078, 20960, 23596, 18364, 17772, 24135, 4151, 12659, 16306, 21607, 6130, 1828, 3814, 4816, 8849, 16171, 16082, 22057, 3161, 20751, 8548, 14287, 12229, 21845, 22344, 9951, 23999, 503, 1622, 4470, 23692, 24198, 4047, 7694, 13444, 6906, 24618, 3022, 3139, 9424, 12640, 21703, 10284, 476, 2702, 5054, 25815, 19949, 22666, 26010, 20945, 4478, 7479, 6536, 14952, 1836, 22517, 2075, 7228, 11652, 19094, 14099, 17154, 9507, 21706, 2004, 3224, 17465, 1953, 19093, 25727, 23673, 23433, 6122, 20013, 24659, 4988, 9517, 12399, 6297, 8784, 19516, 25277, 15162, 16244, 8806, 9286, 5390, 17891, 24601, 17097, 19993, 25002, 12663, 8448, 5466, 7099, 14234, 9614, 8518, 15325, 6555]
        # print(len(exceptresult))
        # for x in exceptresult:
        #     if x not in rec_list_check:
        #         print(x, 'not match!')
        # for x in rec_list_check:
        #     if x not in exceptresult:
        #         print(x, 'not match!')
        print("done!")