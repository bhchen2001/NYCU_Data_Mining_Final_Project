class bookEclat():
    def __init__(self, transactions, min_sup = 0.5):
        self.min_sup = min_sup
        self.transactions = transactions
        self.item_dict = {}
        self.item_list = []
        self.frequent_itemset = []
        self.abs_sup = 0

    # def loadData(self):
    #     with open(self.file_path) as input_file:
    #         for line in input_file:
    #             output_line = re.sub('^[0-9]* [0-9]* [0-9]* ', '', line)
    #             output_line = output_line.strip('\n')
    #             self.transactions.append([s for s in output_line.split(' ')])

    def dataPreprocessing(self):
        # scan the transaction and create dictionary for each item
        for tid in range(0, len(self.transactions)):
            for item in self.transactions[tid]:
                if item in self.item_dict:
                    if tid not in self.item_dict[item]:
                        self.item_dict[item].append(tid)
                else:
                    self.item_dict[item] = [tid]
        # sort the dictionary by item
        self.item_dict = dict(sorted(self.item_dict.items()))
        print(self.item_dict)
        self.abs_sup = self.min_sup * len(self.transactions)
        print("absolute support: ", self.abs_sup)

    def frequentSingle(self):
        # get the frequent itemset with single item
        for item in self.item_dict.keys():
            item_sup = len(self.item_dict[item])
            if item_sup >= self.abs_sup:
                self.frequent_itemset.append([item, item_sup])
                self.item_list.append(item)
        self.item_list = sorted(self.item_list)

    def eclat(self, pre_itemset, pre_transac, last_idx):
        for idx in range(last_idx + 1, len(self.item_list)):
            transaction_union = pre_transac.intersection(self.item_dict[self.item_list[idx]])
            if len(transaction_union) >= self.abs_sup:
                # print("union sup: ", len(transaction_union))
                new_itemset = pre_itemset + [self.item_list[idx]]
                self.frequent_itemset.append([new_itemset, len(transaction_union)])
                self.eclat(new_itemset, transaction_union, idx)

    def process(self):
        # print("transaction: ", self.transactions)
        self.dataPreprocessing()
        self.frequentSingle()
        print("frequent itemset: ", self.frequent_itemset)
        for idx in range(0, len(self.item_list)):
            self.eclat([self.item_list[idx]], set(self.item_dict[self.item_list[idx]]), idx)
        print(len(self.frequent_itemset))