from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pickle

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')



@app.route('/predict', methods=['GET','POST'])

def predict():
    sm_df=pd.read_pickle('dataframe.pkl')
    count = CountVectorizer()
    count_matrix = count.fit_transform(sm_df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    sm_df = sm_df.reset_index()
    titles = sm_df['title']
    indices = pd.Series(sm_df.index, index=sm_df['title'])
    indices.index.astype(str)
    movie='abc'
    list1=[]
    if(request.method=='POST'):
        movie=request.form.get('movie')
        try:
            idx=sm_df[sm_df['title'].str.startswith(movie)].index[0]
        except Exception as e:
            return render_template('nfound.html', movie=movie)
        else:
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            movie_indices = [i[0] for i in sim_scores]
            list1=titles.iloc[movie_indices].tolist()
            print(len(list1))
            return render_template('result.html', list=list1, len=len(list1),  movie=movie)

    if(request.method=='POSTtoHOME'):
        return render_template(home.html)
    
    

if __name__ == '__main__':
	app.run(debug = True)
