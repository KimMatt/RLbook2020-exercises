import pytest 
import pandas as pd
from collections import defaultdict

def dictionary_means(dict_list):
    df = pd.DataFrame(dict_list)
    df_means = df.mean()

    result = defaultdict(float)
    for key, val in zip(df_means.index, df_means[:]):
        result[key] = val
    return df_means


def test_dictionary_means():
    
    # dictionaries
    d1 = {'a':1, 'b':5, 'c':10}
    d2 = {'b': 10, 'c':100, 'a':3}

    answer = defaultdict(float)
    answer['a'] = (1+3)/2
    answer['b'] = (5+10)/2
    answer['c'] = (10+100)/2

    test_answer = dictionary_means([d1, d2])
    
    for key in answer.keys():
        assert answer[key] == test_answer[key]
    