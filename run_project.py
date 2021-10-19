'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from tqdm import tqdm
from preprocessor import Preprocessor
from indexer import Indexer
from collections import OrderedDict
from linkedlist import LinkedList
import inspect as inspector
import sys
import argparse
import json
import time
import math
import random
import flask
from flask import Flask
from flask import request
import hashlib

app = Flask(__name__)


class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()
        self.posting_lists = list()
        self.skip_list = list()
        self.posting_lists_tfidf = list()
        self.tfidf_list = LinkedList()
        self.tfidf_dict = OrderedDict({})

    def _merge(self, l1, l2, skip1, skip2):
        """ Implement the merge algorithm to merge 2 postings list at a time.
            Use appropriate parameters & return types.
            While merging 2 postings list, preserve the maximum tf-idf value of a document.
            To be implemented."""
        comp_ = 0
        count_1 = 0
        count_2 = 0
        merged_list = list()

        post_list = self.indexer.get_index()['random']
        t = post_list.traverse_list()

        l1.sort(reverse=False)
        l2.sort(reverse=False)

        while count_1 < len(l1) and count_2 < len(l2):
            if l1[count_1] == l2[count_2]:
                comp_ = comp_ + 1
                merged_list.append(l1[count_1])
                count_1 = count_1 + 1
                count_2 = count_2 + 1
            elif l1[count_1] < l2[count_2]:
                comp_ = comp_ + 1
                count_1 = count_1 + 1 + skip1

                if count_1 >= len(l1):
                    count_1 = count_1 - skip1
                else:
                    if l1[count_1] > l2[count_2]:
                        count_1 = count_1 - skip1
            else:
                comp_ = comp_ + 1
                count_2 = count_2 + 1 + skip2

                if count_2 >= len(l2):
                    count_2 = count_2 - skip2
                else:
                    if l2[count_2] > l1[count_1]:
                        count_2 = count_2 - skip2
        if merged_list:
            merged_list.sort(reverse=False)

        return merged_list, comp_, t

    def _daat_and(self, skip=False):
        """ Implement the DAAT AND algorithm, which merges the postings list of N query terms.
            Use appropriate parameters & return types.
            To be implemented."""

        if len(self.posting_lists) == 1:
            return self.posting_lists[0], 0

        merged_docs_ = self.posting_lists[0]
        skip1 = self.skip_list[0]
        final_list = list()
        comparisons_ = 0

        for i, _ in enumerate(self.posting_lists[1:]):
            if not skip:
                skip1 = skip2 = 0
            else:
                skip2 = self.skip_list[1:][i]

            merged_list, comp_, temp_ = self._merge(merged_docs_, _, skip1, skip2)
            comparisons_ = comparisons_+comp_
            merged_docs_ = merged_list

            if not merged_docs_:
                return merged_docs_, comparisons_
            else:
                if skip:
                    skip1 = math.floor(math.sqrt(len(merged_docs_)))
                    if skip1 * skip1 == len(merged_docs_):
                        skip1 = skip1 - 1

                for doc_ in merged_docs_:
                    if doc_ not in final_list:
                        final_list.append(doc_)

        if final_list:
            final_list.sort(reverse=False)

        return final_list, comparisons_

    def _daat_and_tfidf(self, list_):
        dict_ = dict()
        for doc_ in list_:
            if doc_ not in dict_:
                dict_[doc_] = list()
            for pos_ in self.posting_lists_tfidf:
                for _ in pos_:
                    if doc_ == _[0]:
                        dict_[doc_].append(_[1])
        for _ in dict_:
            dict_[_] = max(dict_[_])

        new_dict = dict()
        for _ in sorted(dict_.keys()):
            new_dict[_] = dict_[_]

        return sorted(new_dict, key=new_dict.get, reverse=True)

    def _get_postings(self, index, term):
        """ Function to get the postings list of a term from the index.
            Use appropriate parameters & return types.
            To be implemented."""
        return index[term].traverse_list()

    def _output_formatter(self, op):
        """ This formats the result in the required format.
            Do NOT change."""
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
            Already implemented, but you can modify the orchestration, as you seem fit."""
        with open(corpus, 'r', encoding='utf-8') as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()

    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """

        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

    def run_queries(self, query_list, random_command):
        """ DO NOT CHANGE THE output_dict definition"""
        output_dict = {'postingsList': {},
                       'postingsListSkip': {},
                       'daatAnd': {},
                       'daatAndSkip': {},
                       'daatAndTfIdf': {},
                       'daatAndSkipTfIdf': {},
                       'sanity': self.sanity_checker(random_command)}
        index = self.indexer.get_index()

        for query in tqdm(query_list):
            """ Run each query against the index. You should do the following for each query:
                1. Pre-process & tokenize the query.
                2. For each query token, get the postings list & postings list with skip pointers.
                3. Get the DAAT AND query results & number of comparisons with & without skip pointers.
                4. Get the DAAT AND query results & number of comparisons with & without skip pointers, 
                    along with sorting by tf-idf scores."""

            input_term_arr = self.preprocessor.tokenizer(query)

            self.posting_lists = list()
            self.posting_lists_tfidf = list()
            self.skip_list = list()

            for term in input_term_arr:
                if term in list(index.keys()):
                    _, postings = self._get_postings(index, term)
                    index[term].add_skip_connection()
                    skip_postings = index[term].traverse_skips(postings)

                    postings.sort(reverse=False)
                    skip_postings.sort(reverse=False)

                    output_dict['postingsList'][term] = postings
                    output_dict['postingsListSkip'][term] = skip_postings

                    self.posting_lists.append(postings)
                    self.posting_lists_tfidf.append(self.indexer.tfidf_dict[term].traverse_list_tfidf())
                    self.skip_list.append(self.indexer.skips_dict[term])
                else:
                    break

            if self.posting_lists:
                daat, daat_comps = self._daat_and(skip=False)
                daat_skip, daat_skip_comps = self._daat_and(skip=True)
                daat_tfidf = self._daat_and_tfidf(daat)
                daat_skip_tfidf = self._daat_and_tfidf(daat_skip)
            else:
                daat, daat_comps = [], 0
                daat_skip, daat_skip_comps = [], 0
                daat_tfidf = []
                daat_skip_tfidf = []

            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(daat)
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(daat_skip)
            and_op_no_score_no_skip_sorted, and_results_cnt_no_skip_sorted = self._output_formatter(daat_tfidf)
            and_op_no_score_skip_sorted, and_results_cnt_skip_sorted = self._output_formatter(daat_skip_tfidf)

            output_dict['daatAnd'][query.strip()] = {}
            output_dict['daatAnd'][query.strip()]['results'] = and_op_no_score_no_skip
            output_dict['daatAnd'][query.strip()]['num_docs'] = and_results_cnt_no_skip
            output_dict['daatAnd'][query.strip()]['num_comparisons'] = daat_comps

            output_dict['daatAndSkip'][query.strip()] = {}
            output_dict['daatAndSkip'][query.strip()]['results'] = and_op_no_score_skip
            output_dict['daatAndSkip'][query.strip()]['num_docs'] = and_results_cnt_skip
            output_dict['daatAndSkip'][query.strip()]['num_comparisons'] = daat_skip_comps

            output_dict['daatAndTfIdf'][query.strip()] = {}
            output_dict['daatAndTfIdf'][query.strip()]['results'] = and_op_no_score_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_docs'] = and_results_cnt_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_comparisons'] = daat_comps

            output_dict['daatAndSkipTfIdf'][query.strip()] = {}
            output_dict['daatAndSkipTfIdf'][query.strip()]['results'] = and_op_no_score_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_docs'] = and_results_cnt_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_comparisons'] = daat_skip_comps

        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ This function handles the POST request to your endpoint.
        Do NOT change it."""
    start_time = time.time()

    queries = request.json["queries"]
    random_command = request.json["random_command"]

    """ Running the queries against the pre-loaded index. """
    output_dict = runner.run_queries(queries, random_command)

    """ Dumping the results to a JSON file. """
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    """ Driver code for the project, which defines the global variables.
        Do NOT change it."""

    output_location = "data/project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, help="Corpus File name, with path.",
                        default='data/input_corpus.txt')
    parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    parser.add_argument("--username", type=str,
                        help="Your UB username. It's the part of your UB email id before the @buffalo.edu. "
                             "DO NOT pass incorrect value here", default='nirajche')

    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    """ Initialize the project runner"""
    runner = ProjectRunner()

    """ Index the documents from beforehand. When the API endpoint is hit, queries are run against 
        this pre-loaded in memory index. """
    runner.run_indexer(corpus)

    app.run(host="0.0.0.0", port=9999)
