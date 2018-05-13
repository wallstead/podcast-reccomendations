# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from IPython.display import display
import spacy
import string
import os
import pandas as pd
import numpy as np
import sys


def lematize(docs):
    # Remove extra spaces and periods
    no_spaces = docs.replace('...', ' ').replace('....', ' ').replace('  ', ' ').replace('/', ' ').replace('&', 'and').replace("’", "'").replace("!", "").replace("2", "").replace("2007", "").replace("august", "").replace("850", "").replace("650", "").replace("500", "").replace("5", "").replace("340", "").replace("3", "").replace("007", "")

    # Remove punctuation
    punc_list = set(string.punctuation)
    no_punc = ''.join([char for char in no_spaces if char not in punc_list])

    # Remove unicode
    printable = set(string.printable)
    no_uni = ''.join([char for char in no_punc if char in printable])

    # Run the doc through spaCy
    nlp = spacy.load('en')
    spacy_doc = nlp(no_uni)

    # Lemmatize and lower text
    tokens = [token.lemma_.lower() for token in spacy_doc]
    return tokens


if __name__ == '__main__':
    # Setting empty dataframe
    df = pd.DataFrame()

    # Iterating through all tsv files
    path = 'data/df_popular_podcasts.csv'
    # Loading csv file as dataframe
    temp_df = pd.read_csv(path)
    print(temp_df)
    # Concating new tsv file to current df
    df = pd.concat([df, temp_df], axis=0)

    # Setting column names
    df.columns = ['name', 'artwork', 'genre_ids', 'episode_count', 'episode_duration', 'itunes_url',
                  'feed_url', 'podcast_url', 'description']

    # Limiting df to just podcast name & description for now - may add in other features later
    df = df[['feed_url', 'name', 'description']]

    # Sorting by description
    df.sort_values('description', inplace=True)

    # Dropping duplicates by description (some podcasts are listed multiple times by platform)
    df.drop_duplicates('description', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Taking a small sample for testing - will remove later
    to_compare = 11

    df = df.head(to_compare)

    # Adding domain specific stop words - will create a histogram to determine any
    # other frequently occuring words that don't add meaning here
    stop_words = set(list(ENGLISH_STOP_WORDS) +
                     ['podcast', 'show', 'mp3', 'android', 'iphone', 'ipad'])

    # Instantiating tfidf vectorizer
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=lematize)

    # Getting vectors from podcast descriptions
    vectors = vectorizer.fit_transform(df['description'])
    # Changing vectors to a pandas dataframe
    vectors = pd.DataFrame(vectors.todense())
    # Setting the tokens as the column names
    words = vectorizer.get_feature_names()
    vectors.columns = words
    df = pd.concat([df, vectors], axis=1)

    # Compare the documents to themselves; higher numbers are more similar
    # The diagonal is comparing a document to itself, so those are 1's (100% similar)
    cos_sims = linear_kernel(vectors, vectors)
    # Removing 1's on the diagonals
    np.place(cos_sims, cos_sims >= 0.99, 0)

    # let string lengths be as long as they need to be
    pd.set_option('display.max_colwidth', -1)

    # Getting the podcast that is most similar for each podcast
    most_similar = cos_sims.argsort(axis=1)[::-1]

    max_pods_to_recommend = 10

    most_similar_in_order = []

    for rankArr in most_similar:
        most_similar_this_pod_in_order = []
        looking_for = to_compare - 1 # look for top rank first

        while looking_for >= to_compare - max_pods_to_recommend:
            i = 0 # to go through array
            while rankArr[i] != looking_for:
                i = i + 1

            index_of_value_found = i
            value_found = rankArr[index_of_value_found]
            most_similar_this_pod_in_order.append(df['name'][i])
            looking_for = looking_for - 1

        most_similar_in_order.append(most_similar_this_pod_in_order)


    print(most_similar)
    print(most_similar_in_order)
    # df['most_similar'] = [list(df['name'][x]) for x in most_similar]
    df['most_similar'] = most_similar_in_order

    tf = df[['name','feed_url','most_similar']]
    print(tf.to_json(orient='index'))


    # 25 - 15 sec | 50 - 30 sec | 100 - 1 min | 200 - 2 min
