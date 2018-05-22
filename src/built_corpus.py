from gensim.corpora import Dictionary, MmCorpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from datetime import datetime
import pickle


# internal imports
import pre_clean as clean

class Built_Corpus:
    '''
    Note: The functions in the Built_Corpus class are meant to also operate separate
          from the class, ie, not fitting the class variables and only return their
          specific results. The fit method updates all the class variables with
          the returns from each function.
    '''

    def __init__(self, filepaths=None, total_samples=None):
        '''
        Instantiating the object without fitting gives access to the contained
        methods.

        Initializing with a list of filepaths will trigger the .fit() method
        and populate fields.

        To use a sample value different from the default, it can be fed into
        the constructor.  However, if the object is instantiated with a samples
        value but no filepaths list, then the samples field in the object will
        set to the default value in the fit method (all documents) instead of
        the value given to the initializer.

        After object is initialized, call the fit() method to populate
        fields, or use the methods of the object without fitting.
        '''
        self.filepaths = None
        self.total_samples = None
        self.full_corpus_ = None
        self.full_corpus_cleaned_ = None
        self.tf_ = None
        self.tfidf_ = None
        self.tf_vectorizer_ = None
        self.tfidf_vectorizer_ = None
        self.token_lst_ = None
        self.vocabulary_ = None
        self.serialized_text_path = None
        self.full_corpus_outpath = r'data/full_corpus.txt'
        self.full_corpus_cleaned_outpath = r'data/full_corpus_cleaned.txt'
        self.serialized_text_path = 'data/cards_serialized.mm'
        self.vocabulary_outpath = 'data/cards.vocab'

        if filepaths != None and total_samples == None:
            self.fit(filepaths)
        elif filepaths != None and total_samples != None:
            self.fit(filepaths, total_samples)

    def get_fields(self):
        return vars(self)

    def concat_corpora(self, filepaths, full_corpus_outpath, total_samples='all'):
        '''
        Runs through a number of plotting functions and outputs plots for EDA.  The
        random seed is set so that new cards are added incrementally to the corpus
        in order to model a growing corpus of flashcards, which would be the norm.
        Note that if the samples is less than 400, the SVD
        explained variance plots may fail given their current params.

        INPUT:
            filepaths:   list of filepaths to flash card data in question,answer CSVs
            n_samples: number of records to draw from each deck in the filepaths list

        RETURNS:
        '''
        print('Concatenate list of Decks to build corpus')
        start = datetime.now()

        np.random.seed(0)

        # split total samples by the number of decks to pull from
        if total_samples != 'all': n_samples = int(total_samples / len(filepaths))

        # import and clean corpora, add to list of dataframes
        frames = []
        for file in filepaths:
            # TODO: clumsy way to read in entire deck, will fix later
            if total_samples == 'all': n_samples = len(clean.read_cards(file))
            print('... importing {} cards from {}'.format(str(n_samples), file))
            df = clean.read_cards(file)
            rows = np.random.choice(df.index.values, n_samples)
            df = df.iloc[rows]
            frames.append(df)

        print('Compiling cleaned decks, randomizing and saving to file: {}'.format(full_corpus_outpath))
        full_corpus = pd.concat(frames, ignore_index=True)
        np.savetxt(full_corpus_outpath, full_corpus.values, fmt='%s')

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        # update the samples count if all cards are read in
        if total_samples == 'all': self.total_samples = len(full_corpus)
        return full_corpus

    def clean_corpus(self, full_corpus, full_corpus_cleaned_outpath):
        '''
        Runs cleaning functions on the compiled corpus, collapses the questions
        and answers into single strings

        INPUT:  pandas dataframe corpus with "questions" column and "answers" column

        OUTPUT: pandas Series of strings that can be converted to tf and tfidf
        '''
        print('\nCleaning full corpus and converting to Series of strings for vectorization'
            + '\n   - Saving to {}'.format(full_corpus_cleaned_outpath))
        start = datetime.now()

        corpus_clean = clean.clean_dataframe(full_corpus)
        corpus_clean_collapsed = clean.collapse_df(corpus_clean)

        np.savetxt(full_corpus_cleaned_outpath, corpus_clean_collapsed.values, fmt='%s')

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        return corpus_clean_collapsed

    def tf_vectorizer(self, corpus_clean_collapsed):
        '''
        Create TF (for LDA) The fit CountVectorizer is returned
        in order to extract the vocabulary and build a Gensim Dictionary.  A list
        of tokens is extracted from the CountVectorizer in order to later serialize
        the text

        INPUT: pandas series of documents

        OUTPUT: TF sparse matrix
                the fit CountVectorizer

        '''
        print('\nCreate (for LDA) and TF-IDF (for SVD, NMF) from pandas Series of documents'
            + '\nTokenize the clean and collapsed corpus')
        start = datetime.now()

        tf_vectorizer = CountVectorizer(min_df=5, max_df=0.80, stop_words='english',lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        tf = tf_vectorizer.fit_transform(corpus_clean_collapsed)

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        return tf, tf_vectorizer

    def tfidf_vectorizer(self, corpus_clean_collapsed):
        '''
        Create TF-IDF (for SVD, NMF) from pandas Series of documents
        Vectorizer parameters are hardcoded.

        INPUT: pandas series of documents

        OUTPUT: tfidf
                the fit TfidfVectorizer
        '''
        tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        tfidf = tfidf_vectorizer.fit_transform(corpus_clean_collapsed)

        return tfidf, tfidf_vectorizer

    def build_token_list(self, corpus_clean, tf_vectorizer):
        '''
        Create a list of tokens for the documents

        INPUT:  pandas series of documents

        OUTPUT: list of document tokens
        '''
        print('\nCreate token list from cleaned corpus')
        start = datetime.now()
        corpus_clean_list = corpus_clean.tolist()

        word_tokenizer = tf_vectorizer.build_tokenizer()
        token_list = [word_tokenizer(sentence) for sentence in corpus_clean_list]

        end = datetime.now()
        print("   Time taken: {}".format(end - start))
        return token_list

    def compile_gensim_vocab(self, tf_vectorizer, vocabulary_outpath):
        '''
        Extract the vocabulary from fit sklearn count vectorizer, save in Gensim's
        Dictionary format
        '''
        print('\nCreate and Save to file Vocabulary from sklearn CountVectorizer using'
            + 'Gensim Dictionary')
        start = datetime.now()

        sklearn_vocab = tf_vectorizer.vocabulary_

        vocabulary_gensim = {}
        for key, val in sklearn_vocab.items():
            vocabulary_gensim[val] = key
        vocabulary = Dictionary()
        vocabulary.merge_with(vocabulary_gensim)

        vocabulary.save(vocabulary_outpath)

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        return vocabulary

    def serialize_text_to_disk(self, vocabulary, token_lst, serialized_text_path):
        '''
        Serializes text onto disk using Gensim MmCorpus

        INPUT:  gensim vocabulary
                list of tokenized documents

        OUTPUT: filename
        '''
        print('\nSerialize text onto disk using Gensim MmCorpus')
        start = datetime.now()

        MmCorpus.serialize(serialized_text_path, (vocabulary.doc2bow(doc) for doc in token_lst))
        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        return serialized_text_path

    def fit(self, filepaths, total_samples=1200):
        '''
        Runs functions to compile the corpus

        INPUT: filepaths:     list of filename paths, changing the filename list
                              after the object is instantiated will lead to
                              overwriting of class fields and exported files
               total_samples: total number of samples in outputted corpus

        SAVES TO DISK: vocabulary
                       serialized documents

        OUTPUT:
               Updates Object class fields
               pickles and outputs the object
        '''
        print('Running Corpus Compiler from list of data files.')
        start = datetime.now()

        # fill in class fields
        self.filepaths = filepaths
        self.total_samples = total_samples
        self.full_corpus_ = self.concat_corpora(self.filepaths, self.full_corpus_outpath, self.total_samples)
        self.full_corpus_cleaned_ = self.clean_corpus(self.full_corpus_, self.full_corpus_cleaned_outpath)
        self.tf_, self.tf_vectorizer_ = self.tf_vectorizer(self.full_corpus_cleaned_)
        self.tfidf_, self.tfidf_vectorizer_ = self.tfidf_vectorizer(self.full_corpus_cleaned_)
        self.token_lst_ = self.build_token_list(self.full_corpus_cleaned_, self.tf_vectorizer_)
        self.vocabulary_ = self.compile_gensim_vocab(self.tf_vectorizer_, self.vocabulary_outpath)
        self.serialize_text_to_disk(self.vocabulary_, self.token_lst_, self.serialized_text_path)

        # dump pickled object as backup with timestamp
        self.dump_built_corpus()

        end = datetime.now()
        print("\n\n\nCorpus Builder ------ Total Time taken: {}, for {} samples\n\n\n".format(end - start, total_samples))

    def dump_built_corpus(self):
        '''
        Simply dumps the built object.
        '''
        print('Dumping pickled built_corpus object')
        start = datetime.now()

        timestamp = str(datetime.now()).replace(' ', '_').replace(':','')
        pickle_filename = 'data/pickle/built_corpus_' + timestamp + '.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        end = datetime.now()
        print("   Time taken: {}".format(end - start))

if __name__ == '__main__':

    filepaths = ['data/datascience_flashcards.txt', 'data/biology_flashcards.txt', 'data/history_flashcards.txt']

    # built_corpus = Built_Corpus()
    #
    # built_corpus.fit(filepaths, total_samples=1200)

    built_corpus = Built_Corpus(filepaths, total_samples=1200)

    print(built_corpus.filepaths)
    print(built_corpus.total_samples)
    print(len(built_corpus.full_corpus_))
    print(len(built_corpus.full_corpus_cleaned_))
    print(type(built_corpus.tf_), built_corpus.tf_.shape)
    print(type(built_corpus.tfidf_), built_corpus.tfidf_.shape)
    print(type(built_corpus.tf_vectorizer_))
    print(type(built_corpus.tfidf_vectorizer_))
    print(len(built_corpus.token_lst_))
    print(type(built_corpus.vocabulary_))
    print(built_corpus.serialized_text_path)
    print(built_corpus.full_corpus_outpath)
    print(built_corpus.full_corpus_cleaned_outpath)
    print(built_corpus.serialized_text_path)
    print(built_corpus.vocabulary_outpath)
