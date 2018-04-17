import src.pre_clean as clean
import src.wordcloud as wc
import pandas as pd

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    data = 'data/ds_flashcards_2.txt'
    df = clean.read_cards(data)

    df_clean = clean.clean_dataframe(df)

    df_clean['question'][0]

    df_collapsed = clean.collapse_df(df_clean)

    df_collapsed.shape
    type(df_collapsed)

    df_collapsed.str.contains('ttt')
    df_collapsed[0]

    # Create Wordcloud from data with stripped out html
    wc.create_wordcloud_from_df(df_collapsed)





    '''
    Topic Modeling
    '''
    NUM_TOPICS = 5


    vectorizer = TfidfVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform(df_collapsed)

    len(vectorizer.get_feature_names())

    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(data_vectorized)
    print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    print(lda_Z[0])

    # Build a Non-Negative Matrix Factorization Model
    nmf_model = NMF(n_components=NUM_TOPICS)
    nmf_Z = nmf_model.fit_transform(data_vectorized)
    print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

    # Build a Latent Semantic Indexing Model
    lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
    lsi_Z = lsi_model.fit_transform(data_vectorized)
    print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

    # Let's see how the first document in the corpus looks like in different topic spaces
    print(lda_Z[0])
    print(nmf_Z[0])
    print(lsi_Z[0])


    def print_topics(model, vectorizer, top_n=10):
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            print([(vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]])

    print("Latent Dirichlet Allocation Model:")
    print_topics(lda_model, vectorizer)
    print("=" * 20)

    print("NMF Model:")
    print_topics(nmf_model, vectorizer)
    print("=" * 20)

    print("LSI Model:")
    print_topics(lsi_model, vectorizer)
    print("=" * 20)
