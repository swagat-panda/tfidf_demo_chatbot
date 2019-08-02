import numpy as np
import pandas as pd
import string
import pickle
import operator
import matplotlib.pyplot as plt

data_raw = pd.read_csv("./data/chitchat_dataset_salarpuria.csv")
data_raw.head()
from test_tfidf import ChitChat

context_and_target = []
for index, row in data_raw.iterrows():
    context_and_target.append((row['MESSAGE'], row['RESPONSE']))

# print(context_and_target)

count = 0

correct = 0
fail = 0
print("+++++++++++REPORT CHITCHAT++++++++++++++++++++")
chat = ChitChat()
for loop in context_and_target:
    response_primary = chat.enter_query(loop[0].lower())
    if response_primary == loop[1]:
        correct = correct + 1
    else:
        print("FAILED---------->")
        print("question==> ", loop[0])
        print("predicted_response==> ", response_primary)
        print("actual_response==> ", loop[1])
        print("====================================================================================")
        fail = fail + 1
print("================= Final report =============")
print(correct, "correct--------->")
print(fail, "fail------------>")
print(len(context_and_target),"total test cases")
print((correct/len(context_and_target))*100,"== Total accuracy")
