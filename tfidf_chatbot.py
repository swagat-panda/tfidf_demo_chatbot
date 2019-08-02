# _____TF-IDF libraries_____
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# _____helper Libraries_____
import os
import pickle  # would be used for saving temp files
import csv  # used for accessing the dataset
import timeit  # to measure time of training
import random  # used to get a random number


class ChitChat:
    def __init__(self):
        DATA_PATH = os.path.abspath(os.path.dirname(__file__))
        DATA_PATH = os.path.join(DATA_PATH, "data")
        self.file_path = os.path.join(DATA_PATH, "chitchat_dataset_salarpuria_mam .csv")
        self.pickle_vector_path = os.path.join(DATA_PATH, "previous_tfidf_vectorizer.pickle")
        self.training_matrix_path = os.path.join(DATA_PATH, "previous_tfidf_matrix_train.pickle")
        sentences_file = open(self.file_path, "r")

        df = pd.read_csv(self.file_path)
        df = df.dropna()

        self.keywords = df["MESSAGE"].values.astype(str).tolist()
        self.response = df["RESPONSE"].values.astype(str).tolist()

        try:
            # --------------to use------------------#
            f = open(self.pickle_vector_path, 'rb')
            self.tfidf_vectorizer = pickle.load(f)
            f.close()

            f = open(self.training_matrix_path, 'rb')
            self.tfidf_matrix_train = pickle.load(f)
            f.close()
            # ----------------------------------------#
        except:
            # ---------------to train------------------#
            self.tfidf_vectorizer, self.tfidf_matrix_train = self.training_model(self.file_path,
                                                                                 self.pickle_vector_path,
                                                                                 self.training_matrix_path)
            # -----------------------------------------#

    def training_model(self, file_path, pickle_vector_path, training_matrix_path):
        i = 0
        sentences = []
        sentences.append(" eos")
        sentences.append(" eos")

        for row in self.keywords:
            sentences.append(row)
            i += 1

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)

        f = open(pickle_vector_path, 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()

        f = open(training_matrix_path, 'wb')
        pickle.dump(tfidf_matrix_train, f)
        f.close()

        return tfidf_vectorizer, tfidf_matrix_train

    def predict_response(self, test_set_sentence, minimum_score):
        # enter your test sentence
        test_set = (test_set_sentence, "")
        # use the learnt dimension space
        # to run TF-IDF on the query
        tfidf_matrix_test = self.tfidf_vectorizer.transform(test_set)
        # then run cosine similarity between the 2 tf-idfs
        cosine = cosine_similarity(tfidf_matrix_test, self.tfidf_matrix_train)
        cosine = np.delete(cosine, 0)

        # then get the max score
        max = cosine.max()
        # response_index = 0

        # if score is more than minimum_score
        if (max > minimum_score):
            # we can afford to get multiple high score documents to choose from
            new_max = max - 0.01
            # load them to a list
            list = np.where(cosine > new_max)
            # choose a random one to return to the user
            # this happens to make Lina answers diffrently to same sentence
            response_index = random.choice(list[0])

        else:
            # else we would simply return the highest score
            return "", 0
        j = 0

        # loop to return the next cell on the row , ( the response cell )
        for row in self.response:
            j += 1  # we begin with 1 not 0 &    j is initialized by 0
            if j == response_index:
                return row, max

    def enter_query(self, query):
        minimum_score = 0.8
        query_response, score = self.predict_response(query, minimum_score)
        return query_response


if __name__ == '__main__':
    chat = ChitChat()
    while True:
        user_response = input("Me : ")
        user_response = user_response.lower()
        if user_response == "quit":
            break
        response_primary = chat.enter_query(user_response)
        print("marvin: ", response_primary)
