'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from linkedlist import LinkedList
from collections import OrderedDict


class Indexer:
    def __init__(self):
        """ Add more attributes if needed"""
        self.inverted_index = OrderedDict({})
        self.tfidf_dict = OrderedDict({})
        self.skips_dict = OrderedDict({})

    def get_index(self):
        """ Function to get the index.
            Already implemented."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index
            Already implemented."""
        for t in tokenized_document:
            self.add_to_index(t, doc_id)

    def add_to_index(self, term_, doc_id_):
        """ This function adds each term & document id to the index.
            If a term is not present in the index, then add the term to the index & initialize a new postings list (linked list).
            If a term is present, then add the document to the appropriate position in the postings list of the term.
            To be implemented."""
        if term_ not in list(self.inverted_index.keys()):
            self.inverted_index[term_] = LinkedList()
        self.inverted_index[term_].insert_at_end(int(doc_id_))

    def sort_terms(self):
        """ Sorting the index by terms.
            Already implemented."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """ For each postings list in the index, add skip pointers.
            To be implemented."""
        for term_ in list(self.inverted_index.keys()):
            self.inverted_index[term_].traverse_list()
            self.skips_dict[term_] = self.inverted_index[term_].add_skip_connection()

    def calculate_tf_idf(self):
        """ Calculate tf-idf score for each document in the postings lists of the index.
            To be implemented."""
        dict_tfidf = dict()

        for term_ in list(self.inverted_index.keys()):
            pos_list, pos_list_unique = self.inverted_index[term_].traverse_list()
            tf_list = list()

            for _ in pos_list_unique:
                tf_list.append(pos_list.count(_))

            dict_tfidf[term_] = dict(zip(pos_list_unique, tf_list))

        unique_docs = list()
        for dict_ in list(dict_tfidf.values()):
            for _ in dict_:
                if _ not in unique_docs:
                    unique_docs.append(_)

        total_token_dict = {_: 0 for _ in unique_docs}
        for dict_ in list(dict_tfidf.values()):
            for _ in dict_:
                total_token_dict[_] = total_token_dict[_] + dict_[_]

        dict_post = dict()
        for term_ in dict_tfidf:
            dict_post[term_] = list()

            for _ in dict_tfidf[term_]:
                dict_post[term_].append([(dict_tfidf[term_][_]/total_token_dict[_])*(
                        len(unique_docs)/len(list(dict_tfidf[term_].keys()))), _])

        for term_ in dict_post:
            self.tfidf_dict[term_] = LinkedList()
            [self.tfidf_dict[term_].insert_at_end_tfidf(_) for _ in dict_post[term_]]
