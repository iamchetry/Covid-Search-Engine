'''
@author: Sougata Saha
Institute: University at Buffalo
'''

import math


class Node:

    def __init__(self, value=None, doc=None, next=None):
        """ Class to define the structure of each node in a linked list (postings list).
            Value: document id, Next: Pointer to the next node
            Add more parameters if needed.
            Hint: You may want to define skip pointers & appropriate score calculation here"""
        self.value = value
        self.doc = doc
        self.next = next


class LinkedList:
    """ Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'
        Each term in the inverted index has an associated linked list object.
        Feel free to add additional functions to this class."""
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        if self.start_node is None:
            return
        else:
            n = self.start_node
            while n is not None:
                traversal.append(n.value)
                n = n.next

            uniques_ = list()
            for _ in traversal:
                if _ not in uniques_:
                    uniques_.append(_)

            self.length = len(uniques_)
            return traversal, uniques_

    def traverse_skips(self, uniques_):
        if len(uniques_) <= 2:
            return list()
        return uniques_[0:len(uniques_):self.n_skips+1]

    def traverse_list_tfidf(self):
        traversal = []
        if self.start_node is None:
            print("List has no element")
            return
        else:
            n = self.start_node
            cnt = 1
            while n is not None:
                traversal.append([n.doc, n.value])
                cnt += 1
                n = n.next
            return traversal

    def add_skip_connection(self):
        self.n_skips = math.floor(math.sqrt(self.length))
        if self.n_skips * self.n_skips == self.length:
            self.n_skips = self.n_skips - 1
        return self.n_skips

    def insert_at_end(self, value):
        """ Write logic to add new elements to the linked list.
            Insert the element at an appropriate position, such that elements to the left are lower than the inserted
            element, and elements to the right are greater than the inserted element.
            To be implemented. """
        new_node = Node(value=value)
        n = self.start_node

        if self.start_node is None:
            self.start_node = new_node
            self.end_node = new_node
            return

        elif self.start_node.value >= value:
            self.start_node = new_node
            self.start_node.next = n
            return

        elif self.end_node.value <= value:
            self.end_node.next = new_node
            self.end_node = new_node
            return

        else:
            while n.value < value < self.end_node.value and n.next is not None:
                n = n.next

            m = self.start_node
            while m.next != n and m.next is not None:
                m = m.next
            m.next = new_node
            new_node.next = n
            return

    def insert_at_end_tfidf(self, params):
        value, doc = params
        new_node = Node(value=value, doc=doc)
        n = self.start_node

        if self.start_node is None:
            self.start_node = new_node
            self.end_node = new_node
            return

        elif self.start_node.value <= value:
            self.start_node = new_node
            self.start_node.next = n
            return

        elif self.end_node.value >= value:
            self.end_node.next = new_node
            self.end_node = new_node
            return

        else:
            while n.value > value > self.end_node.value and n.next is not None:
                n = n.next

            m = self.start_node
            while m.next != n and m.next is not None:
                m = m.next
            m.next = new_node
            new_node.next = n
            return

