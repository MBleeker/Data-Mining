# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.

"""
Interface to query data with functionality for reading queries from svmlight
format, both sequentially and in batch mode.
"""

import sys
import gc
import gzip
import numpy as np
import os.path
import pandas as pd

class Query:
    __qid__ = None
    __feature_vectors__ = None
    __labels__ = None
    __predictions__ = None
    __comments__ = None
    # document ids will be initialized as zero-based, os they can be used to
    # retrieve labels, predictions, and feature vectors
    __docids__ = None
    __ideal__ = None
    
    def __hash__(self):
        return self.__qid__.__hash__()
    
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__qid__ == other.__qid__)

    def __init__(self, qid, feature_vectors, labels=None, comments=None):
        self.__qid__ = qid
        self.__feature_vectors__ = feature_vectors
        self.__labels__ = np.asarray(labels,dtype="float32")
        self.__comments__ = comments

    def has_ideal(self):
        return not self.__ideal__ is None

    def set_ideal(self, ideal):
        self.__ideal__ = ideal

    def get_ideal(self):
        return self.__ideal__

    def get_qid(self):
        return self.__qid__

    def get_docids(self):
        return self.__docids__

    def get_document_count(self):
        return len(self.__docids__)

    def get_feature_vectors(self):
        return self.__feature_vectors__

    def set_feature_vector(self, docid, feature_vector):
        self.__feature_vectors__[docid.get_id()] = feature_vector

    def get_feature_vector(self, docid):
        return self.__feature_vectors__[docid.get_id()]

    def get_labels(self):
        return self.__labels__

    def set_label(self, docid, label):
        self.__labels__[docid.get_id()] = label

    def get_label(self, docid):
        return self.__labels__[docid.get_id()]
    
    def get_labelById(self, docid):
        return self.__labels__[docid]

    def set_labels(self, labels):
        self.__labels__ = labels

    def get_comments(self):
        return self.__comments__

    def get_comment(self, docid):
        if self.__comments__ is not None:
            return self.__comments__[docid.get_id()]
        return None

    def get_predictions(self):
        return self.__predictions__

    def get_prediction(self, docid):
        if self.__predictions__:
            return self.__predictions__[docid.get_id()]
        return None

    def set_predictions(self, predictions):
        self.__predictions__ = predictions

    def write_to(self, fh, sparse=False):
        for doc in self.__docids__:
            features = [':'.join((repr(pos + 1),
                repr(value))) for pos, value in enumerate(
                self.get_feature_vector(doc)) if not (value == 0 and sparse)]
            print >> fh, self.get_label(doc), ':'.join(("qid",
                self.get_qid())), ' '.join(features),
            comment = self.get_comment(doc)
            if comment:
                print >> fh, comment
            else:
                print >> fh, ""


class QueryStream:
    """iterate over a stream of queries, only keeping one query at a time"""
    __reader__ = None
    __numFeatures__ = 0

    def __init__(self, X, y, qry_ids):
    # qry_ids should be the 'srch_id' column
        self.X = X
        self.y = y
        self.qry_ids = qry_ids
        self.qry_ids_unique = pd.unique(qry_ids)
        self.__num_features__ = X.shape[1]
        self.count = 0
    def __iter__(self):
        return self

    # with some inspiration from multiclass.py
    # http://svmlight.joachims.org/svm_struct.html
    # takes self and number of expected features
    # returns qid and features, one query at a time
    def next(self):
        if self.count >= len(self.qry_ids_unique):
            raise StopIteration
        qid_current = self.qry_ids_unique[self.count]
        X_this_qry = self.X[self.qry_ids==qid_current,:]
        y_this_qry = self.y[self.qry_ids==qid_current]
        self.count += 1
        return Query(qid_current, X_this_qry, y_this_qry)

    # read all queries from a file at once
    def read_all(self, print_stuff = False):
        queries = {}
        index = 0
        for query in self:
            if print_stuff:            
                if index % 100 == 0:
                    print "loaded ",index
            index += 1
            queries[query.get_qid()] = query
        return queries


class Queries:
    """a list of queries with some convenience functions"""

    __num_features__ = 0
    __queries__ = None

    # cache immutable query values
    __qids__ = None
    __feature_vectors__ = None
    __labels__ = None

    def __init__(self, X, y, qry_ids):
        # X is a numpy array with just the features
        # y are the target relevance labels
        # qry_ids are the srch_id for every row in X (will contain duplicates)
        self.__queries__ = QueryStream(X, y, qry_ids).read_all()
        self.__num_features__ = X.shape[1]

    def __iter__(self):
        return iter(self.__queries__.itervalues())

    def __getitem__(self, index):
        return self.get_query(index)

    def __len__(self):
        return len(self.__queries__)

    def keys(self):
        return self.__queries__.keys()

    def values(self):
        return self.__queries__.values()

    def get_query(self, index):
        return self.__queries__[index]

    def get_qids(self):
        if not self.__qids__:
            self.__qids__ = [query.get_qid() for query in self]
        return self.__qids__

    def get_labels(self):
        if not self.__labels__:
            self.__labels__ = [query.get_labels() for query in self]
        return self.__labels__

    def get_feature_vectors(self):
        if not self.__feature_vectors__:
            self.__feature_vectors__ = [query.get_feature_vectors()
                for query in self]
        return self.__feature_vectors__

    def set_predictions(self):
        raise NotImplementedError("Not yet implemented")

    def get_predictions(self):
        if not self.__predictions__:
            self.__predictions__ = [query.get_predictions() for query in self]
        return self.__predictions__

    def get_size(self):
        return self.__len__()