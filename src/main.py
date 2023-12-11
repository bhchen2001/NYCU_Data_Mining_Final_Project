from recommend_knn import bookKNN
from user_based_CF import UserBasedCF
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--data_path", dest="data_path",
                      help="data path", metavar="FILE")
    parser.add_option("-m", "--method", dest="method",
                      help="method for recommendation")
    parser.add_option("-k", "--k", dest="k")
    (options, args) = parser.parse_args()

    data_path = str(options.data_path) + '/'
    if options.method == 'knn':
        my_knn = bookKNN(data_path, k = 3)
        my_knn.process()

    if options.method == 'user_based_cf':
        my_user_based_cf = UserBasedCF(data_path, "cosine_similarity", k = options.k)
        my_user_based_cf.process()
