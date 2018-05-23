import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import six
from datetime import datetime
import pyLDAvis.gensim
from matplotlib.patches import Rectangle
from sklearn.decomposition import NMF, TruncatedSVD
import warnings
warnings.filterwarnings('ignore')


def plot_mds_dist(names, pos):
    xs, ys = pos[:, 0], pos[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot()

    for x, y, name in zip(xs, ys, names):
        if names[0] == name:
            color = 'orange'
        elif names[1] == name:
            color = 'red'
        else:
            color = 'green'


        ax = plt.scatter(x, y, c=color)
        ax = plt.text(x, y, name)
    plt.savefig('temp_mds2d.png')
    plt.show()
    plt.close()


def plot_mds_dist3d(names, pos):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

    for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
        ax.text(x, y, z, s)

    plt.savefig('temp_mds3d.png')
    plt.show()
    plt.close()

def plot_top_words_table(data, num_topics, document_count,
                         lda_gensim_coherence,
                         col_width=2.0,
                         row_height=0.625,
                         font_size=14,
                         header_color='#40466e',
                         row_colors=['#f1f1f2', 'w'],
                         edge_color='w',
                         bbox=[0, 0, 1, 1],
                         header_columns=0,
                         ax=None,
                         **kwargs
                         ):
    '''
    Outputs a table of the top words discerned by the LDA
    model.
    '''
    print('\nGenerating top words table at:')
    start = datetime.now()

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

    print('            ' + filename)

    end = datetime.now()
    print("   Time taken: {}".format(end - start))


def plot_pyLDAvis(lda_model, corpus_cards, vocabulary, n_topics):
    '''
    Saves a Gensim pyLDAvis plot as html.
        - Using tSNE for projection
        - R is set to 15, to display 15 terms
    '''
    print('\nGenerating pyLDAvis Gensim (on tf) plot at:')
    start = datetime.now()

    pyl_data_cv = pyLDAvis.gensim.prepare(lda_model, corpus_cards, vocabulary, R=15, mds='tsne')
    filename = 'images/pyldavis/lda_vect_topics_' + str(len(corpus_cards)) + '_docs_' + str(n_topics) + '_topics.html'
    pyLDAvis.save_html(pyl_data_cv, filename)

    end = datetime.now()
    print('            ' + filename)
    print("   Time taken: {}".format(end - start))



def plot_nmf_reconstruction(tfidf, solver='mu', max_iter=100):
    '''
    Plot nmf_reconstruction plots for ranges 1-10 and
       10-100 by 10s
    '''
    print('\nGenerating sklearn NMF plot on tfidf to look for elbows on n topics for best reconstruction:')
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

    n_topics = [[1, 2, 3, 4, 5, 6,7,8,9,10],
                [10,20,30,40,50,60,70,80,90,100]]



    for range in n_topics:

        reconstruction_error = []

        for topicnum in range:
            nmf = NMF(n_components=topicnum, max_iter=max_iter, solver=solver, init="random", random_state=0, beta_loss='kullback-leibler')
            nmf.fit(tfidf, max_iter)
            reconstruction_error.append(nmf.reconstruction_err_)

        fig = plt.figure()

        plt.style.use('bmh')
        fig.suptitle('NMF Reconstruction Error vs. Number of Topics\n' + '(With {} Iterations) on tfidf'.format(max_iter), fontdict=font)

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)

        ax.set_xlabel('Number of Topics')
        ax.set_ylabel('NMF Reconstruction Error')

        ax.scatter(range, reconstruction_error)
        randcards = r'{}'.format(tfidf.shape[0])
        solver_name = 'Multiplicative Update (mu)'
        loss_function = 'Kullback-Leibler Divergence'

        extra = Rectangle((0, 0), 1, 1, fc='w', fill=False, edgecolor='none', linewidth=0)

        keytext = 'Sample Size: {}\nSolver: {}\nLoss Function: {}'.format(randcards, solver_name, loss_function)

        ax.text(0.45, 0.92, keytext, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='darkseagreen', alpha=1), fontdict=font2)

        # plt.show()
        filename = 'images/nmf_reconstr/nmf_reconstr_' + str(range[-1]) + 'components_' + str(tfidf.shape[0]) + '_docs.png'
        fig.savefig(filename, dpi=fig.dpi)

        print('            ' + filename)
        print('  saving {}'.format(filename))
        plt.close()

if __name__ == '__main__':
    pass
