from recommend_knn import bookKNN
from user_based_cf import UserBasedCF
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--data_path", dest="data_path", default = "../dataset", help="data path", metavar="FILE")
    parser.add_option("-m", "--method", dest="method", help="method for recommendation: 'knn' or 'user_based_cf'")
    parser.add_option("-k", "--k", dest="k", default = 10, type = int, help="k nearest readers")
    parser.add_option("-r", "--recommend_num", dest="recommend_num", default = 20, type = int, help="number of recommended books")
    (options, args) = parser.parse_args()

    if options.method is None:
        parser.error("method is required: 'knn' or 'user_based_cf'")

    data_path = str(options.data_path) + '/'
    if options.method == 'knn':
        my_knn = bookKNN(data_path, k = options.k, recommend_num=options.recommend_num)
        my_knn.process()
    elif options.method == 'user_based_cf':
        my_user_based_cf = UserBasedCF(data_path, "cosine_similarity", k = options.k, recommend_num=options.recommend_num)
        my_user_based_cf.process()
