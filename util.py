import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from gensim.models.wrappers import LdaMallet
from sklearn.feature_extraction.text import CountVectorizer 
from gensim import corpora, models, matutils
from gensim.test.utils import datapath

def read_crowdtangle_file(file):    
    df = pd.read_csv(file, dtype='unicode')
    """
    Extract and clean text fields from a Crowd Tangle Data File.
    """
    # Extract text from all fields with text
    df['Text']= df['Message'].map(str) + ' '     \
                + df['Image Text'].map(str)+ ' ' \
                + df['Link Text'].map(str) + ' ' \
                + df['Description'].map(str)
    
    # Remove extraneous characters
    df['Text'] = (df['Text'].str.replace('nan',' ')
                            .str.replace('\n',' ')
                            .str.replace('\r',' ')
                            .str.replace(r"http\S+",' '))
                            #.str.decode('ascii','ignore')
    df = df[['Facebook Id','Text']]
    return df

def read_crowdtangle_files(file_paths):
    """
    Read multiple csv files using multi-threading.
    """
    with ThreadPoolExecutor() as executor:
        with tqdm(total = len(file_paths)) as progress:
            futures = []
            for file in file_paths:
                future = executor.submit(read_crowdtangle_file, file)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
            return(results)

def create_corpus(text, stop_words):
    """Helper transform text into a dictionary and bag of words corpus.
    
    Args:
        text: Series of text documents
        stop_words: List of words to omit from the analysis 
        
    Returns:
        A dictionary and corpus
    """  
    vectorizer = CountVectorizer(stop_words=stop_words, 
                                 ngram_range = (1,2), 
                                 token_pattern="\\b[a-z][a-z][a-z]+\\b",
                                 max_df=0.9, min_df=5,
                                 max_features=1000000) 
    vectorizer.fit(text)
    doc_word = vectorizer.transform(text).transpose()
    corpus = matutils.Sparse2Corpus(doc_word)
    
    word2id = dict((v, k) for v, k in vectorizer.vocabulary_.items())
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

    dictionary = corpora.Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = word2id
    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    return dictionary, corpus
