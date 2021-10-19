'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from nltk.stem import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        """ Splits each line of the document, into doc_id & text.
            Already implemented"""
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        """ Implement logic to pre-process & tokenize document text.
            Write the code in such a way that it can be re-used for processing the user's query.
            To be implemented."""
        ps = PorterStemmer()
        text = text.lower()
        text = re.sub('[^a-zA-Z0-9 ]', ' ', text).strip()
        text = [_ for _ in text.split() if _ not in ['', ' ']]
        text = [ps.stem(_) for _ in text if _ not in stopwords.words('english', 'spanish')]

        return text
