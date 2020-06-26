from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pickle

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


m_df=pd.read_csv("movies_metadata.csv")
m_df['genres'] = m_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
small_mdf = pd.read_csv('links_small.csv')
small_mdf = small_mdf[small_mdf['tmdbId'].notnull()]['tmdbId'].astype('int')
credit_df=pd.read_csv('credits.csv')
keyword_df=pd.read_csv('keywords.csv')
m_df.drop(19730,inplace=True)
m_df.drop(29503,inplace=True)
m_df.drop(35587,inplace=True)
keyword_df['id'] = keyword_df['id'].astype('int')
credit_df['id'] = credit_df['id'].astype('int')
m_df['id'] = m_df['id'].astype('int')
m_df = m_df.merge(credit_df, on='id')
m_df = m_df.merge(keyword_df, on='id')
sm_df = m_df[m_df['id'].isin(small_mdf)]

sm_df['cast'] = sm_df['cast'].apply(literal_eval)
sm_df['crew'] = sm_df['crew'].apply(literal_eval)
sm_df['keywords'] = sm_df['keywords'].apply(literal_eval)
sm_df['cast_size'] = sm_df['cast'].apply(lambda x: len(x))
sm_df['crew_size'] = sm_df['crew'].apply(lambda x: len(x))

sm_df['director'] = sm_df['crew'].apply(get_director)

sm_df['cast'] = sm_df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
sm_df['cast'] = sm_df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

sm_df['keywords'] = sm_df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

sm_df['cast'] = sm_df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

sm_df['director'] = sm_df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
sm_df['director'] = sm_df['director'].apply(lambda x: [x,x])

s = sm_df.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

s = s.value_counts()
s = s[s > 1]

stemmer = SnowballStemmer('english')


sm_df['keywords'] = sm_df['keywords'].apply(filter_keywords)
sm_df['keywords'] = sm_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
sm_df['keywords'] = sm_df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

sm_df['soup'] = sm_df['keywords'] + sm_df['cast'] + sm_df['director'] + sm_df['genres']
sm_df['soup'] = sm_df['soup'].apply(lambda x: ' '.join(x))
sm_df.to_pickle('dataframe.pkl')
