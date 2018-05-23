# external imports
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary, MmCorpus
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from pandas.tools.plotting import table
import numpy as np
from datetime import datetime
import pickle


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
        self.lda_coherence_model = None
        self.lda_coherence_score = None

        # compile the data for use in the class
        self.built_corpus = build.Built_Corpus(self.filepaths, total_samples=1200)

        # pull filepath out from the serialized corpus
        self.corpus_cards = MmCorpus(self.built_corpus.serialized_text_path)

        self.fit()

    def fit(self):
        '''
        Populate the Class fields for the cards object
        '''
        print('\nPopulating fields in the LdaFlashcards object ')
        start = datetime.now()

        self.fit_lda()
        self.set_coherence()
        self.dump_lda_flashcards()

        end = datetime.now()
        print("\n\n\nLdaFlashcards Built ------ Total Time taken: {}, for {} samples\n\n\n".format(end - start, self.built_corpus.total_samples))

    def fit_lda(self):
        '''
        Read in serialized cards from disk. Fit the LdaModel for the class.

        Only operates on class fields.
        '''
        print('\nRun Gensim LdaModel on serialized documents')
        start = datetime.now()

        # Feed params from built_corpus into the LDA model
        self.lda_model = LdaModel(corpus=self.corpus_cards, num_topics=self.n_topics, id2word=self.built_corpus.vocabulary_, distributed=False, chunksize=2000, passes=10, update_every=1, alpha='symmetric', eta=None, decay=0.7, offset=10.0, eval_every=10, iterations=self.max_iter, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None)

        end = datetime.now()
        print("   Time taken: {} on {} topics, max iterations: {}".format(end - start, self.n_topics, self.max_iter))

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

    def set_coherence(self):
        '''
        Use the Gensim CoherenceModel to gauge the internal coherence of the
        model
        '''
        print('\nDetermining Coherence measure from fit model.')
        start = datetime.now()

        self.lda_coherence_model = CoherenceModel(model=self.lda_model, corpus=self.corpus_cards, dictionary=self.built_corpus.vocabulary_, coherence='u_mass')

        self.lda_coherence_score = float(str('{0:.2f}'.format(self.lda_coherence_model.get_coherence())))

        print('   ** Coherence Score for {} topics and {} cards: {} **'.format(self.n_topics, self.built_corpus.total_samples, self.lda_coherence_score))

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

    def get_best_n_topics(self):
        '''
        maybe this can be fitting svd on the tfidf, then running a 'grid search' on kmeans with the svd output.  Reducing Sum of Squares
            Sum of Squares: ((between cluster scatter matrix) / (within cluster scatter matrix)) * k
        '''
        print('TODO: make this return number of kmeans clusters optimizing for Sum of Squares')
        pass

    def get_top_words(self, num_words=10):
        '''
        Get a dataframe of the top words per topic

        INPUT:  number of words per topic

        OUTPUT: returns pandas dataframe
        '''
        print('\nGetting Pandas dataframe of top words per topic')
        start = datetime.now()

        top_topics_df = pd.DataFrame([[word for rank, (word, prob) in enumerate(words)]
                  for topic_id, words in self.lda_model.show_topics(formatted=False, num_words=num_words, num_topics=self.n_topics)])

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        return top_topics_df


    def build_plots(self, plots=[]):
        '''
        Pass in a list of strings of the types of plots to be built.  Read
        through the function for the names available.
        '''
        print('\nCreating Plots: '.format(plots))
        start = datetime.now()

        if 'top_words_table' in plots:
            pt.plot_top_words_table(self.get_top_words(), self.n_topics, self.built_corpus.total_samples, self.lda_coherence_score)

        if 'pyLDAvis' in plots:
            pt.plot_pyLDAvis(self.lda_model, self.corpus_cards,
                             self.built_corpus.vocabulary_, self.n_topics)

        if 'nmf_reconstruction' in plots:
            pt.plot_nmf_reconstruction(self.built_corpus.tfidf_)

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

    def dump_lda_flashcards(self):
        '''
        Simply dumps the built object.
            - TODO: this is redundantly dumping the built corpus, but for now
                    we'll do this in order to track down the weird
        '''
        print('\nDumping pickled lda_flashcards object')
        start = datetime.now()

        timestamp = str(datetime.now()).replace(' ', '_').replace(':','')
        pickle_filename = 'data/pickle/lda_flashcards/lda_flashcards_model_' + timestamp + '.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

if __name__ == '__main__':

    filepaths = ['data/datascience_flashcards.txt', 'data/biology_flashcards.txt', 'data/history_flashcards.txt']

    cards = LdaFlashcards(filepaths=filepaths, n_topics=3, total_samples='all')

    cards.build_plots(['nmf_reconstruction'])
