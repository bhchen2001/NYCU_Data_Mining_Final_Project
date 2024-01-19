import numpy as np
import sys
import os
import re
from optparse import OptionParser
sys.path.insert(1, '../../src/FIM')


import collections
import math
import itertools

import pandas as pd


def itemsets_transformation(df: pd.DataFrame) -> tuple:

    assert len(df) != 0, "Dataframe is empty"
    itemsets = df.values
    single_items = np.array(df.columns)

    return itemsets, single_items


def itemset_optimisation(
    itemsets: np.array,
    single_items: np.array,
    minsup: int,
) -> tuple:
    """
    Downward-closure property of H-Mine algorithm.
        Optimizes the itemsets matrix by removing items that do not
        meet the minimum support.

    Args:
        itemsets (np.array): matrix of bools or binary
        single_items (np.array): array of single items
        minsup (int): minimum absolute support

    Returns:
        itemsets (np.array): reduced itemsets matrix of bools or binary
        single_items (np.array): reduced array of single items
        single_items_support (np.array): reduced single items support
    """

    single_items_support = np.array(np.sum(itemsets, axis=0)).reshape(-1)
    items = np.nonzero(single_items_support >= minsup)[0]
    itemsets = itemsets[:, items]
    single_items = single_items[items]
    single_items_support = single_items_support[items]

    return itemsets, single_items, single_items_support


class TransactionEncoder():
    def __init__(self) -> None:
        pass

    def fit(self, itemsets: list) -> object:
        """
        This method creates a list of unique items in the dataset.

        Args:
            itemsets (list): dataset
        """
        unique_items = []
        for transaction in itemsets:
            for item in transaction:
                if item not in unique_items:
                    unique_items.append(item)
        self.columns = sorted(unique_items)
        self.columns_dict = {item: idx for idx, item in enumerate(self.columns)}
        return self

    def transform(self, itemsets: list, set_pandas=True) -> np.array:
        """
        This method converts the dataset into a binary matrix.

        Args:
            itemsets (list): dataset
        """
        output = np.zeros((len(itemsets), len(self.columns)), dtype=bool)
        for idx, transaction in enumerate(itemsets):
            for item in transaction:
                if item in self.columns_dict:
                    output[idx, self.columns_dict[item]] = True
        if set_pandas:
            return pd.DataFrame(output, columns=self.columns)
        return output

    def inverse_transform(self, itemsets: list) -> list:
        """
        This method converts the binary matrix into a dataset.

        Args:
            itemsets (list): binary matrix
        """
        output = []
        for transaction in itemsets:
            tmp = []
            for idx, item in enumerate(transaction):
                if item:
                    tmp.append(self.columns[idx])
            output.append(tmp)
        return output

    def fit_transform(self, itemsets: list, set_pandas=True) -> np.array:
        """
        This method combines fit and transform methods.

        Args:
            itemsets (list): dataset
        """
        return self.fit(itemsets).transform(itemsets, set_pandas=set_pandas)

class bookFPgrowth():

    def __init__(self, transactions):
        self.transactions = transactions

    def load_data(self):
        te = TransactionEncoder()
        df = te.fit_transform(self.transactions, set_pandas=True)

        return df

    def fpgrowth(self, df,
                min_support: float = 0.5,
                show_colnames: bool = True,
                max_len: int = None
                ) -> pd.DataFrame:

        if min_support <= 0.0:
            raise ValueError(
                "`min_support` must be a positive "
                "number within the interval `(0, 1)`. "
                "Got %s." % min_support
            )
        minsupp = math.ceil(min_support * len(df))

        fptree, _ = self.construct_fptree(df, minsupp)
        self.generator = self.fpgrowth_driver(fptree, minsupp, max_len)
        res_df = self.generate_fis(self.generator, df.shape[0])

        if show_colnames:
            mapping = {idx: item for idx, item in enumerate(df.columns)}
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([mapping[i] for i in x])
            )

        return res_df


    def fpgrowth_driver(self, tree, minsupp, max_len):
        count = 0
        items = tree.nodes.keys()
        if tree.is_path():
            size_remain = len(items) + 1
            if max_len:
                size_remain = max_len - len(tree.conditional_items) + 1
            for i in range(1, size_remain):
                for transaction in itertools.combinations(items, i):
                    count += 1
                    support = min([tree.nodes[i][0].count for i in transaction])
                    yield support, tree.conditional_items + list(transaction)

        elif not max_len or len(tree.conditional_items) < max_len:
            for item in items:
                count += 1
                support = sum([node.count for node in tree.nodes[item]])
                yield support, tree.conditional_items + [item]

        if not tree.is_path() and (not max_len or len(tree.conditional_items) < max_len):
            for item in items:
                conditional_tree = tree.conditional_tree(item, minsupp)
                for sup, iset in self.fpgrowth_driver(conditional_tree, minsupp, max_len):
                    yield sup, iset


    def construct_fptree(self, df, minsupp):
        num_of_transactions = df.shape[0]
        # print("num_of_transactions: ", num_of_transactions)

        itemsets, _ = itemsets_transformation(df)
        # print("itemsets: ", itemsets)

        item_support = np.array(np.sum(itemsets, axis=0))
        item_support = item_support.reshape(-1)
        items = np.nonzero(item_support >= minsupp)[0]
        # print("items: ", items)

        indices = np.argsort(item_support[items])
        rank = {item: i for i, item in enumerate(items[indices])}
        # print("indices: ", indices)

        fptree = FPTree(rank)
        for i in range(num_of_transactions):
            non_null = np.where(itemsets[i, :])[0]
            transaction = [item for item in non_null if item in rank]
            transaction.sort(key=rank.get, reverse=True)
            fptree.add_transaction(transaction)

        return fptree, rank


    def generate_fis(self, enerator, num_of_transaction):
        FREQUENTITEMSETS = {}
        for sup, items in self.generator:
            FREQUENTITEMSETS[frozenset(items)] = sup / num_of_transaction

        res_df = pd.DataFrame([FREQUENTITEMSETS.values(), FREQUENTITEMSETS.keys()]).T
        res_df.columns = ["support", "itemsets"]
        return res_df
    
    def process(self):
        df = self.load_data()
        self.fpgrowth(df, min_support = 0.001)


class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = collections.defaultdict(list)
        self.conditional_items = []
        self.rank = rank

    def conditional_tree(self, conditional_item: str or int, minsupp: int):
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[conditional_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count
        items = [item for item in count if count[item] >= minsupp]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}
        conditional_tree = FPTree(rank)

        if len(items) > 0:
            for idx, branch in enumerate(branches):
                branch = sorted([i for i in branch if i in rank], key=rank.get, reverse=True)
                conditional_tree.add_transaction(branch, self.nodes[conditional_item][idx].count)
            conditional_tree.conditional_items = self.conditional_items + [conditional_item]
        return conditional_tree

    def add_transaction(self, transaction, count=1):
        self.root.count += count
        if len(transaction) == 0:
            return
        index = 0
        node = self.root
        for item in transaction:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break
        for item in transaction[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True


class FPNode(object):
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)

    def itempath_from_root(self):
        path = []
        if self.item is None:
            return path
        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent
        path.reverse()
        return path



def data_preprocessing(file_path):
    data = []
    # modification for input file format
    with open(file_path) as input_file:
        for line in input_file:
            output_line = re.sub('^[0-9]* [0-9]* [0-9]* ', '', line)
            output_line = output_line.strip('\n')
            data.append([s for s in output_line.split(' ')])
    te = TransactionEncoder()
    df = te.fit_transform(data, set_pandas=True)

    return df

# if __name__ == "__main__":

#     optparser = OptionParser()
#     optparser.add_option(
#         "-f", "--inputFile", dest="input", help="filename containing csv", default='A.csv'
#     )
#     optparser.add_option(
#         "-s",
#         "--minSupport",
#         dest="minS",
#         help="minimum support value",
#         default=0.1,
#         type="float",
#     )

#     (options, args) = optparser.parse_args()

#     fileName = None
#     if options.input is None:
#         fileName = sys.stdin
#     elif options.input is not None:
#         fileName = options.input
#     else:
#         print("No dataset filename specified, system with exit\n")
#         sys.exit("System will exit")

#     minSupport = options.minS
#     case = os.path.basename(options.input)
#     case = case.split('.')[0]
#     exe_time = 0

#     df = data_preprocessing(fileName)
#     result = fpgrowth(df, min_support = minSupport)

#     sorted_df = result.sort_values(by='support', ascending=False)
#     sorted_df['support'] = sorted_df['support'].apply(lambda x: round(x * 100, 1))

#    # 格式化 'itemsets' 列为 {} 样式
#     sorted_df['itemsets'] = sorted_df['itemsets'].apply(lambda x: f'{{{", ".join(map(str, x))}}}')

#     # 保存 DataFrame 到文件
#     sorted_df.to_csv('output.txt', sep='\t', index=False, header=False)

#     print(sorted_df)