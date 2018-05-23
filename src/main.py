import seaborn as sns
import src.models as models
import src.pre_clean as clean
import src.wordclouder as wc
import src.plotting as pt
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import nltk
import os
import logging
import pyLDAvis.gensim
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from datetime import datetime

from gensim.models.coherencemodel import CoherenceModel
# from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
import pyLDAvis.sklearn
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus
import spacy
import six


if __name__ == '__main__':

    # Our data
    filenames = ['data/biology_flashcards.txt',
                 'data/datascience_flashcards.txt',
                 'data/history_flashcards.txt']

    # filenames_clean = ['data/biology_flashcards_cleaned.txt',
    #              'data/datascience_flashcards_cleaned.txt',
    #              'data/history_flashcards_cleaned.txt']
    #
    # names = ['Biology', 'Data Science', 'History']
    # names_clean = ['Biology (clean)', 'Data Science (clean)', 'History (clean)']
    #
    # vectorizer = CountVectorizer(input='filename')
    # vectorizer_clean = CountVectorizer(input='filename')
    #
    # # get sparse matrix of file word counts
    # dtm = vectorizer.fit_transform(filenames)
    # dtm_clean = vectorizer_clean.fit_transform(filenames_clean)
    #
    # # build a list of the document vocabularies
    # #    - note, this vocab is built but contains
    # #      a lot of garbage
    # vocab = vectorizer.get_feature_names()
    # len(vocab) # 34152 total words for all, raw data
    # vocab[32000] # ultraviolet
    # # clean vocab
    # vocab_clean = vectorizer_clean.get_feature_names()
    # len(vocab_clean) # 29257 total words for data after cleaning
    #
    # # to get # of occurrences of a term, need to convert from
    # #    sparse matrix to numpy matrix
    # dtm_array = dtm.toarray()
    # # convert vocab to np.array
    # vocab = np.array(vocab)
    #
    # # Look at the number of occurrences of a few words in the raw data
    # #   - distribution
    # dtm_array[0, vocab=='distribution'] # appears 39 times biology
    # dtm_array[1, vocab=='distribution'] # appears 799 times in datascience
    # dtm_array[2, vocab=='distribution'] # appears 19 times in history
    # #   - cell
    # dtm_array[0, vocab=='cell'] # appears 1672 times biology
    # dtm_array[1, vocab=='cell'] # appears 62 times in datascience
    # dtm_array[2, vocab=='cell'] # appears 0 times in history
    # #   - war
    # dtm_array[0, vocab=='war'] # appears 0 times biology
    # dtm_array[1, vocab=='war'] # appears 0 times in datascience
    # dtm_array[2, vocab=='war'] # appears 1019 times in history
    # #   - cosine
    # dtm_array[0, vocab=='cosine'] # appears 0 times biology
    # dtm_array[1, vocab=='cosine'] # appears 8 times in datascience
    # dtm_array[2, vocab=='cosine'] # appears 0 times in history
    # #   - whether
    # dtm_array[0, vocab=='whether'] # appears 11 times biology
    # dtm_array[1, vocab=='whether'] # appears 175 times in datascience
    # dtm_array[2, vocab=='whether'] # appears 10 times in history
    #
    #
    # '''Measuring distance between each flash card deck'''
    # # Euclidean distance
    # eucl_dist = euclidean_distances(dtm)
    # np.round(eucl_dist, 1)
    # # Cosine Similarity/Distance
    # cos_sim = cosine_similarity(dtm)
    # np.round(cos_sim, 2)
    #
    # # Euclidean distance clean
    # eucl_dist_clean = euclidean_distances(dtm_clean)
    # np.round(eucl_dist_clean, 1)
    # # Cosine Similarity/Distance clean
    # cos_sim_clean = cosine_similarity(dtm_clean)
    # np.round(cos_sim_clean, 3)
    #
    #
    #
    #
    #
    #
    # ''' Multidimensional Scaling (MDS) '''
    # # two components as we're plotting points in a two-dimensional plane
    # # "precomputed" because we provide a distance matrix
    # # we will also specify `random_state` so the plot is reproducible.
    #
    # # MDS for euclidean distance raw
    # mds_eucl = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # pos_eucl = mds_eucl.fit_transform(eucl_dist)  # shape (n_components, n_samples)
    #
    # # MDS for euclidean distance clean
    # mds_eucl_clean = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # pos_eucl_clean = mds_eucl_clean.fit_transform(eucl_dist_clean)  # shape (n_components, n_samples)
    # pt.plot_mds_dist(names, pos_eucl_clean)
    #
    # # MDS for cosine similarity raw
    # mds_cos = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # pos_cos = mds_cos.fit_transform(cos_sim)  # shape (n_components, n_samples)
    #
    # pt.plot_mds_dist(names, pos_cos)
    #
    # # MDS for cosine similarity clean
    # mds_cos_clean = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # pos_cos_clean = mds_cos_clean.fit_transform(cos_sim_clean)  # shape (n_components, n_samples)
    #
    # pt.plot_mds_dist(names, pos_cos_clean)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # ''' For generating wordclouds on all data '''
    # # datascience cards
    # data = 'data/datascience_flashcards.txt'
    # df_datascience = clean.read_cards(data)
    # df_datascience_clean = clean.clean_dataframe(df_datascience)
    # df_datascience = clean.collapse_df(df_datascience_clean)
    # df_datascience[79]
    # datascience_tfidf, datascience_count = models.get_vectorizers(df_datascience)
    # # Export datascience cleaned
    # np.savetxt(r'data/datascience_flashcards_cleaned.txt', df_datascience.values, fmt='%s')
    # wc.create_wordcloud_from_df(df_datascience, "/Users/tbot/Dropbox/galvanize/a-smarter-flashcard/images/brain_template.png")
    #
    # # biology cards
    # data = 'data/biology_flashcards.txt'
    # df_biology = clean.read_cards(data)
    # df_biology_clean = clean.clean_dataframe(df_biology)
    # df_biology = clean.collapse_df(df_biology_clean)
    # df_biology[79]
    # np.savetxt(r'data/biology_flashcards_cleaned.txt', df_biology.values, fmt='%s')
    # biology_tfidf, biology_count = models.get_vectorizers(df_biology)
    #
    # wc.create_wordcloud_from_df(df_biology, "/Users/tbot/Dropbox/galvanize/a-smarter-flashcard/images/beaker_template.png")
    #
    # # history cards
    # data = 'data/history_flashcards.txt'
    # df_history = clean.read_cards(data)
    # df_history_clean = clean.clean_dataframe(df_history)
    # df_history = clean.collapse_df(df_history_clean)
    # df_history[79]
    # np.savetxt(r'data/history_flashcards_cleaned.txt', df_history.values, fmt='%s')
    # history_tfidf, history_count = models.get_vectorizers(df_history)
    # wc.create_wordcloud_from_df(df_history, "/Users/tbot/Dropbox/galvanize/a-smarter-flashcard/images/knight_template.png")
    #
    #
    #
    sample_list = [4000]



    for n_samples in sample_list:
        ''' Topic Modeling with pyLDAvis'''

        # set random seed for predictable results
        np.random.seed(0)

        # n_samples = 500
        # compile single corpora
        #   datascience cards
        filename = 'data/datascience_flashcards.txt'
        df = clean.read_cards(filename)
        rows = np.random.choice(df.index.values, n_samples) #len(df))
        df_datascience = df.ix[rows]

        #   biology cards
        filename = 'data/biology_flashcards.txt'
        df = clean.read_cards(filename)
        rows = np.random.choice(df.index.values, n_samples) #len(df))
        df_biology = df.ix[rows]

        #   history cards
        filename = 'data/history_flashcards.txt'
        df = clean.read_cards(filename)
        rows = np.random.choice(df.index.values, n_samples) #len(df))
        df_history = df.ix[rows]

        len(df_datascience)
        len(df_biology)
        len(df_history)

        frames = [df_datascience, df_biology, df_history]

        # Condense read cards into one dataframe, reindex
        full_demo_corpus_filepath = r'data/full_demo_corpus.txt'
        print('Compiling cleaned decks, randomizing and saving to file: {}'.format(full_corpus_filepath))
        corpus = pd.concat(frames, ignore_index=True)
        # rows = np.random.choice(corpus.index.values, len(corpus))
        # corpus = df.ix[rows]
        np.savetxt(full_corpus_filepath, corpus.values, fmt='%s')

        document_count = len(corpus)



        # clean corpus
        start = datetime.now()
        print('\nClean/Collapse Corpus')
        start = datetime.now()
        corpus_clean = clean.clean_dataframe(corpus)
        corpus_collapsed = clean.collapse_df(corpus_clean)
        corpus_collapsed_lst = corpus_collapsed.tolist()
        end = datetime.now()
        print("   Time taken: {}".format(end - start))

        # save out full file
        print('\nSaving cleaned corpus')
        np.savetxt(r'data/full_corpus_cleaned.txt', corpus_collapsed.values, fmt='%s')


        # Use gensim to create Dictionary using the collapsed corpus
        from gensim.corpora import Dictionary, MmCorpus


        print('\nCreate TF-IDF from Collapsed Corpus')
        start = datetime.now()
        tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        corpus_tfidf = tfidf_vectorizer.fit_transform(corpus_collapsed)
        print("   Time taken: {}".format(end - start))


        print('\nCreate TF from Collapsed Corpus')
        start = datetime.now()
        count_vectorizer = CountVectorizer(min_df=5, max_df=0.80, stop_words='english',lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        corpus_tf = count_vectorizer.fit_transform(corpus_collapsed)
        word_tokenizer = count_vectorizer.build_tokenizer()
        token_lst = [word_tokenizer(sentence) for sentence in corpus_collapsed_lst]




        sklearn_vocab = count_vectorizer.vocabulary_
        print("   Time taken: {}".format(end - start))

        print('\nCreate and Save Vocabulary from sklearn vocab using Gensim Dictionary')
        start = datetime.now()
        vocabulary_gensim = {}
        for key, val in sklearn_vocab.items():
            vocabulary_gensim[val] = key
        vocabulary = Dictionary()
        vocabulary.merge_with(vocabulary_gensim)
        vocabulary.save('data/cards.vocab')
        end = datetime.now()
        print("   Time taken: {}".format(end - start))


        print('\nSerialize text onto disk using Gensim MmCorpus')
        start = datetime.now()
        MmCorpus.serialize('data/cards_serialized.mm', (vocabulary.doc2bow(doc) for doc in token_lst))
        end = datetime.now()
        print("   Time taken: {}".format(end - start))







        from gensim.models import LdaModel
        import pyLDAvis as ldavis
        import pyLDAvis.gensim


        print('\nRun Gensim LdaModel on serialized documents')
        start = datetime.now()
        num_topics = 5
        iterations = 10
        # since cards are stored to disk, can read them in using MmCorpus
        corpus_cards = MmCorpus('data/cards_serialized.mm')
         # LdaModel(corpus=corpus_cards, id2word=vocabulary, num_topics=num_topics, chunksize=128, iterations=iterations, alpha='auto')
        lda_cards = LdaModel(corpus=corpus_cards, num_topics=num_topics, id2word=vocabulary, distributed=False, chunksize=2000, passes=10, update_every=1, alpha='symmetric', eta=None, decay=0.7, offset=10.0, eval_every=10, iterations=iterations, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None)

        end = datetime.now()
        print("   Time taken: {} on {} topics, max iterations: {}".format(end - start, num_topics, iterations))


        '''
        Explore topic coherence
        '''

        from gensim.models import CoherenceModel

        cm = CoherenceModel(model=lda_cards, corpus=corpus_cards, dictionary=vocabulary, coherence='u_mass')
        lda_gensim_coherence = str('{0:.2f}'.format(cm.get_coherence()))
        print(lda_gensim_coherence)


        def get_topic_table(lda_cards, num_topics, num_words=6):
            from pandas.tools.plotting import table
            top_topics_df = pd.DataFrame([[word for rank, (word, prob) in enumerate(words)]
                      for topic_id, words in lda_cards.show_topics(formatted=False, num_words=6, num_topics=num_topics)])

            return top_topics_df

        topic_table = get_topic_table(lda_cards, num_topics, num_words=6)


        def render_word_table(data, num_topics, document_count, lda_gensim_coherence, col_width=3.0, row_height=0.625, font_size=14,
                             header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                             bbox=[0, 0, 1, 1], header_columns=0,
                             ax=None, **kwargs):
            if ax is None:
                size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
                fig, ax = plt.subplots(figsize=size)
                ax.axis('off')

            mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

            plt.title('LDA: Top words in {} Topics, {} documents    UMass Coherence: {} '.format(num_topics, document_count, lda_gensim_coherence))

            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(font_size)

            for k, cell in six.iteritems(mpl_table._cells):
                cell.set_edgecolor(edge_color)
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color='w')
                    cell.set_facecolor(header_color)
                else:
                    cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

            filename = 'images/topic_tables/table_docs_' + str(document_count) + '_' + str(num_topics) + '_topics.png'
            plt.savefig(filename)

            plt.close()

        render_word_table(topic_table, num_topics, document_count, lda_gensim_coherence, header_columns=0, col_width=2.0)





        print('\nRun sklearn LDA on corpus in memory')
        start = datetime.now()
        num_topics = 3

        lda_corpus_count = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=None, topic_word_prior=None, learning_method=None, learning_decay=0.7, learning_offset=10.0, max_iter=iterations, batch_size=128, evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=1, verbose=0, random_state=0, n_topics=None)

        lda_corpus_count.fit(corpus_tf)
        end = datetime.now()
        print("   Time taken: {} on {} topics, max iterations: {}".format(end - start, num_topics, iterations))



        print('\nCreate Gensim pyLDAvis plot, save to file')
        start = datetime.now()
        pyl_data_cv = pyLDAvis.gensim.prepare(lda_cards, corpus_cards, vocabulary, R=15)
        pyLDAvis.save_html(pyl_data_cv, 'images/pyldavis/all_count_vect_topics_' + '_docs_' +  str(document_count) + '_' + str(num_topics) + '_topics' + '_gensim.html')
        end = datetime.now()
        print("   Time taken: {}".format(end - start))


        print('\nCreate sklearn pyLDAvis plot, save to file')
        start = datetime.now()
        pyl_data_cv = pyLDAvis.sklearn.prepare(lda_corpus_count, corpus_tf, count_vectorizer, R=15)
        filename = 'images/pyldavis/all_count_vect_topics_'+ '_docs_' +  str(document_count) + '_' + str(num_topics) + '_topics' + '_sklearn.html'
        pyLDAvis.save_html(pyl_data_cv, filename)
        end = datetime.now()
        print("   Time taken: {}".format(end - start))





        print('\nUse sklearn NMF plotting to get an idea of n topics for best reconstruction')
        start = datetime.now()

        plt.style.use('bmh')

        font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }
        font2 = {'family': 'serif',
            'color':  'white',
            'weight': 'normal',
            'size': 8,
            }

        from matplotlib.patches import Rectangle


        def plot_nmf_topic_reconstruction(termfreq, num_topics, tf_type, solver='mu', max_iter=100):


            reconstruction_error = []

            for topicnum in num_topics:
                nmf = NMF(n_components=topicnum, max_iter=max_iter, solver=solver, init="random", random_state=0, beta_loss='kullback-leibler')
                nmf.fit(termfreq, max_iter)
                reconstruction_error.append(nmf.reconstruction_err_)

            fig = plt.figure()

            plt.style.use('bmh')
            fig.suptitle('NMF Reconstruction Error vs. Number of Topics\n' + '(With {} Iterations) on {}'.format(max_iter, tf_type), fontdict=font)

            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.85)

            ax.set_xlabel('Number of Topics')
            ax.set_ylabel('NMF Reconstruction Error')

            ax.scatter(num_topics, reconstruction_error)
            randcards = r'{}'.format(termfreq.shape[0])
            solver = 'Multiplicative Update (mu)'
            loss_function = 'Kullback-Leibler Divergence'

            extra = Rectangle((0, 0), 1, 1, fc='w', fill=False, edgecolor='none', linewidth=0)

            keytext = 'Sample Size: {}\nSolver: {}\nLoss Function: {}'.format(randcards, solver, loss_function)

            ax.text(0.45, 0.92, keytext, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='darkseagreen', alpha=1), fontdict=font2)

            # plt.show()
            filename = 'images/nmf_reconstr/nmf_reconstr_' + str(num_topics[-1]) + 'components_' + tf_type + '_docs_' + str(document_count) +'.png'
            fig.savefig(filename, dpi=fig.dpi)
            print('  saving {}'.format(filename))
            plt.close()

        # topics_10 = [1, 2, 3, 4, 5, 6,7,8,9,10]
        # topics_100 = [10,20,30,40,50,60,70,80,90,100]
        # tf_type = 'tfidf'
        # plot_nmf_topic_reconstruction(corpus_tfidf, topics_10, tf_type)
        # plot_nmf_topic_reconstruction(corpus_tfidf, topics_100, tf_type)
        #
        # tf_type = 'tf'
        # plot_nmf_topic_reconstruction(corpus_tf, topics_10, tf_type)
        # plot_nmf_topic_reconstruction(corpus_tf, topics_100, tf_type)
        # end = datetime.now()
        # print("   Time taken: {}".format(end - start))



'''
        print('\nUse sklearn LSA (TruncatedSVD) plotting to get an idea of n topics for maximizing explained variance')
        start = datetime.now()


        def plot_lsa_explained_variance(termfreq, num_topics, tf_type, max_iter=10):

            explained_variances = []

            for topicnum in num_topics:
                svd = TruncatedSVD(n_components=topicnum, n_iter=max_iter, random_state=0)
                svd.fit(termfreq)

                explained_variances.append(svd.explained_variance_ratio_.sum())



            print(explained_variances)


            fig = plt.figure()
            plt.style.use('bmh')
            fig.suptitle('LSA (TruncatedSVD) Explained Variance vs. Number of Topics\n' + '(With {} Iterations) on {}'.format(max_iter, tf_type), fontdict=font)

            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.85)

            ax.set_xlabel('Number of Topics')
            ax.set_ylabel('Sum Explained Variance Ratio')

            ax.scatter(num_topics, explained_variances)

            randcards = r'{}'.format(termfreq.shape[0])
            alg = 'Fast Randomized SVD'


            keytext = 'Sample Size: {}\nAlgorithm: {}\n'.format(randcards, alg)

            ax.text(0.15, 0.92, keytext, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='darkseagreen', alpha=1), fontdict=font2)


            filename = 'images/lsa_explained_variance/lsa_explained_var' + str(num_topics[-1]) + 'components_' + tf_type + '_docs_' + str(document_count) +'.png'
            plt.savefig(filename, dpi=fig.dpi)
            plt.close()
            print('  saving {}'.format(filename))



        # if n_samples > 1000:
        #     topics_10 = [1, 2, 3, 4, 5, 6,7,8,9, 10]
        #     topics_100 = [10,20,30,40,50,60,70,80,90,100]
        #     plot_lsa_explained_variance(corpus_tfidf, topics_10, tf_type='tfidf')
        #     plot_lsa_explained_variance(corpus_tf, topics_10, tf_type='tf')
        #
        #     plot_lsa_explained_variance(corpus_tfidf, topics_100, tf_type='tfidf')
        #     plot_lsa_explained_variance(corpus_tf, topics_100, tf_type='tf')
        #     end = datetime.now()






        print('\nUse sklearn tSNE to visualize potential topical clusters')
        start = datetime.now()

        from sklearn.manifold import TSNE
        from yellowbrick.text import TSNEVisualizer
        from sklearn.cluster import KMeans
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        def plot_tsne_clusters(termfreq, num_clusters=3):
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            svd = TruncatedSVD(n_components=50, n_iter=10, random_state=0)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            reduced = lsa.fit_transform(termfreq)

            # next, apply kmeans to the corpus to get labels
            clusters = KMeans(n_clusters=num_clusters, init='k-means++')
            clusters.fit(reduced)

            tsne = TSNEVisualizer(decompose=None)
            tsne.fit(reduced, ["cluster {}".format(c) for c in clusters.labels_])

            tsne.finalize()
            filename = r'images/tsne_projections/tSNE_wKMeans_SVD_' + str(num_clusters) + 'clusters' + '_docs_' + str(document_count) + '.png'
            plt.savefig(filename)
            plt.close()


        # plot_tsne_clusters(corpus_tfidf, num_clusters=3)
        # plot_tsne_clusters(corpus_tfidf, num_clusters=5)
        # plot_tsne_clusters(corpus_tfidf, num_clusters=7)
        # plot_tsne_clusters(corpus_tfidf, num_clusters=9)
        # plot_tsne_clusters(corpus_tfidf, num_clusters=11)

        print("   Time taken: {}".format(end - start))


        print('\nUse kmeans silhoutte score and distortion to make elbow plots')
        # The best silhouette value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
        start = datetime.now()

        from sklearn.cluster import KMeans
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from yellowbrick.cluster import KElbowVisualizer

        def kmeans_elbow_plots(termfreq, low=2, high=20, metric='distortion'):
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            svd = TruncatedSVD(n_components=50, n_iter=10, random_state=0)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            reduced = lsa.fit_transform(termfreq)


            # Instantiate the clustering model and visualizer
            visualizer = KElbowVisualizer(KMeans(init='k-means++'), k=(low,high), metric=metric)

            visualizer.fit(reduced) # Fit the training data to the visualizer
            visualizer.finalize()

            filename = r'images/elbow_plots/kmeans_elbow_plot_' + metric + '_'+ str(low) + 'to' + str(high) + '_docs_' + str(document_count) +'.png'
            plt.savefig(filename)
            plt.close()


        # kmeans_elbow_plots(corpus_tfidf, low=2, high=20, metric='silhouette')
        # kmeans_elbow_plots(corpus_tfidf, low=2, high=20, metric='distortion')
        # kmeans_elbow_plots(corpus_tfidf, low=20, high=40, metric='silhouette')
        # kmeans_elbow_plots(corpus_tfidf, low=20, high=40, metric='distortion')

        print("   Time taken: {}".format(end - start))



        print('\nUse kmeans silhouette score to visualize Silhouette Coefficients')
        # The best silhouette value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
        start = datetime.now()

        from sklearn.cluster import KMeans
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from yellowbrick.cluster import SilhouetteVisualizer

        def kmeans_silhouette_plots(termfreq, num_clusters=3):
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            svd = TruncatedSVD(n_components=50, n_iter=10, random_state=0)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            reduced = lsa.fit_transform(termfreq)

            model = KMeans(n_clusters=num_clusters, init='k-means++')
            # Instantiate the clustering model and visualizer
            visualizer = SilhouetteVisualizer(model)

            visualizer.fit(reduced) # Fit the training data to the visualizer
            visualizer.finalize()

            filename = r'images/silhouette_plots/kmeans_silhouette_plot_clusters-' + str(num_clusters) + '_docs_' + str(document_count) +  '.png'
            plt.savefig(filename)
            plt.close()


        # kmeans_silhouette_plots(corpus_tfidf, num_clusters=3)
        # kmeans_silhouette_plots(corpus_tfidf, num_clusters=5)
        # kmeans_silhouette_plots(corpus_tfidf, num_clusters=7)
        # kmeans_silhouette_plots(corpus_tfidf, num_clusters=9)
        # kmeans_silhouette_plots(corpus_tfidf, num_clusters=11)

        print("   Time taken: {}".format(end - start))







##------------------------------------------------------------------------##



# pyLDAvis.save_html(pyl_data_cv, "../images/all_count_vect_topics.html")

    # # Convert corpus to tf-idf and izer
    # #   - For pyLDAvis, need to keep the vectorizers
    # tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    #
    # count_vectorizer = CountVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    #
    # corpus_tfidf = tfidf_vectorizer.fit_transform(corpus_collapsed)
    # corpus_count = count_vectorizer.fit_transform(corpus_collapsed)
    #
    # corpus_tfidf.shape # (36188, 9436)
    #
    # # Create wordmap from entire corpus
    # # wc.create_wordcloud_from_df(corpus_collapsed)
    #
    # ''' Fit Latent Dirichlet Allocation Models '''
    # # With Count Vector, running with default 10 iterations
    # lda_corpus_count = LatentDirichletAllocation(n_topics=3, random_state=0)
    # lda_corpus_count.fit(corpus_count)
    #
    # # With TF-IDF matrix, running with default 10 iterations
    # lda_corpus_tfidf = LatentDirichletAllocation(n_topics=3, random_state=0)
    # lda_corpus_tfidf.fit(corpus_tfidf)
    #
    # # not working in hydrogen
    # # pyLDAvis.sklearn.prepare(lda_corpus_count, corpus_count, count_vectorizer)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # '''
    # DATA CLEANING Dev
    # '''
    # data = 'data/datascience_flashcards.txt'
    # df = clean.read_cards(data)
    # df['question'][79]
    #
    # df_clean = clean.clean_dataframe(df)
    #
    # # df_clean.tail()
    #
    # df_collapsed = clean.collapse_df(df_clean)
    # df['question'][79]
    # df_collapsed[79]
    # df_collapsed.shape
    # type(df_collapsed)
    # df.shape
    #
    # # # looking for occurrence of 'ttt' string
    # ttt = df_collapsed.str.contains('ttt')
    # len(ttt[ttt==True])
    # # ttt[ttt==True]
    # # df_collapsed[333]
    # # df_collapsed[79]
    #
    # # # Create Wordcloud from data with stripped out html
    # # wc.create_wordcloud_from_df(df_collapsed)
    #
    #
    # # df_collapsed.isnull().sum()
    # # ## There are 110 NaN values after cleaning, solved!
    #
    # # Create series as mask for nan values
    # nulls = pd.isnull(df_collapsed)
    # # nulls[nulls == True].index[0]
    # # df_collapsed[79]
    # # df['question'][79]
    #
    #
    #
    #
    # '''
    # Trying out different Topic Modeling alg's
    # '''
    # NUM_TOPICS = 5
    #
    # tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    #
    # count_vectorizer = CountVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    #
    # data_tfidf_vectorized = tfidf_vectorizer.fit_transform(df_collapsed)
    # # feature_names = tfidf_vectorizer.get_feature_names()
    # data_count_vectorized = count_vectorizer.fit_transform(df_collapsed)
    #
    # type(data_count_vectorized)
    #
    # # print(feature_names[8])
    #
    # # Build a Latent Dirichlet Allocation Model
    # lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
    # lda_Z = lda_model.fit_transform(data_count_vectorized)
    # print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    # print(lda_Z[0])
    #
    # # Build a Non-Negative Matrix Factorization Model
    # nmf_model = NMF(n_components=NUM_TOPICS)
    # nmf_Z = nmf_model.fit_transform(data_tfidf_vectorized)
    # print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    #
    # # Build a Latent Semantic Indexing Model
    # lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
    # lsi_Z = lsi_model.fit_transform(data_tfidf_vectorized)
    # print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    #
    # # Let's see how the first document in the corpus looks like in different topic spaces
    # print(lda_Z[0])
    # print(nmf_Z[0])
    # print(lsi_Z[0])
    #
    #
    # def print_topics(model, vectorizer, top_n=10):
    #     for idx, topic in enumerate(model.components_):
    #         print("Topic %d:" % (idx))
    #         print([(vectorizer.get_feature_names()[i], topic[i])
    #                         for i in topic.argsort()[:-top_n - 1:-1]])
    #
    # print("Latent Dirichlet Allocation Model:")
    # print_topics(lda_model, vectorizer)
    # print("=" * 20)
    #
    # print("NMF (Non-negative matrix factorization) Model:")
    # print_topics(nmf_model, vectorizer)
    # print("=" * 20)
    #
    # print("LSI Model:")
    # print_topics(lsi_model, vectorizer)
    # print("=" * 20)
    #
    # df.shape
    # df_collapsed.shape
