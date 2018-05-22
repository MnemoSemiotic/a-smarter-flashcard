# external imports
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from datetime import datetime

# internal imports
import pre_clean as clean
import plotting as pt
import built_corpus as build


class LdaFlashcards(object):
    '''
    This class object currently relies on the Built_Corpus class
    to get it's data
    '''
    def __init__(self, filepaths,
                       n_topics=3,
                       random_seed=True,
                       max_iter=10,
                       total_samples='all'
                ):
        # set random seed for more deterministic behavior
        if random_seed==True: np.random.seed(0)

        self.filepaths = filepaths
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.lda_model = None
        self.coherence = None

        # compile the data for use in the class
        self.built_corpus = build.Built_Corpus(self.filepaths, total_samples=1200)

        self.fit()

    def fit(self):
        '''
        Populate the Class fields for the cards object
        '''


    def fit_lda(self):
        '''
        Read in serialized cards from disk. Fit the LdaModel for the class.

        Only operates on class fields.
        '''
        print('\nRun Gensim LdaModel on serialized documents')
        start = datetime.now()

        corpus_cards = MmCorpus(self.built_corpus.serialized_text_path)

        # Feed params from built_corpus into the LDA model
        self.lda_model = LdaModel(corpus=corpus_cards, num_topics=self.n_topics, id2word=self.built_corpus.vocabulary_, distributed=False, chunksize=2000, passes=10, update_every=1, alpha='symmetric', eta=None, decay=0.7, offset=10.0, eval_every=10, iterations=self.max_iter, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None)

        end = datetime.now()
        print("   Time taken: {} on {} topics, max iterations: {}".format(end - start, self.n_topics, self.max_iter))

    def get_best_n_topics(self):
        '''
        maybe this can be fitting svd on the tfidf, then running a 'grid search' on kmeans with the svd output.  Reducing Sum of Squares
            Sum of Squares: ((between cluster scatter matrix) / (within cluster scatter matrix)) * k
        '''
        print('TODO: make this return number of kmeans clusters optimizing for Sum of Squares')
        pass

    def dump_lda_flashcards(self):
        '''
        Simply dumps the built object.
            - TODO: this is redundantly dumping the built corpus, but for now
                    we'll do this in order to track down the weird
        '''
        print('Dumping pickled lda_flashcards object')
        start = datetime.now()

        timestamp = str(datetime.now()).replace(' ', '_').replace(':','')
        pickle_filename = 'data/lda_flashcards/lda_flashcards' + timestamp + '.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

if __name__ == '__main__':

    filepaths = ['data/datascience_flashcards.txt', 'data/biology_flashcards.txt', 'data/history_flashcards.txt']

    cards = LdaFlashcards(filepaths=filepaths, n_topics=3, total_samples='all')
