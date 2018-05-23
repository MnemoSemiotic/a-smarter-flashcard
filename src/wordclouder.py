import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import path
import random
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import pre_clean as clean

def create_wordcloud_from_df(df, template_filename=None, output_filename='temp_wc'):
    '''
    INPUT: pandas.core.series.Series

    OUTPUT: returns NONE
          : Calls create_wordcloud and generates text from
            pandas series
    '''

    list_of_strings = [str(i) for i in df]

    if filename==None:
        create_wordcloud(' '.join(list_of_strings), output_filename)
    else:
        create_wordcloud_custom(' '.join(list_of_strings), template_filename, output_filename)

def create_wordcloud(text, output_filename):
    '''
    INPUT: string

    OUTPUT: returns NONE
          : saves a figure of a wordcloud in working directory
    '''
    wordcloud = WordCloud().generate(text)

    dpi=300

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(12, 8),dpi=dpi, facecolor='w', edgecolor='k')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    filename = output_filename + '.png'
    plt.savefig(filename, dpi=dpi)


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def create_wordcloud_custom(text, template_filename):
    """
    Using custom colors
    ===================
    Using the recolor method and custom coloring functions.
    """

    d = path.dirname(__file__)

    # read image from filename
    mask = np.array(Image.open(path.join(d, template_filename)))

    wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, margin=1, random_state=1, collocations=False).generate(text)
    # store default colored image
    default_colors = wc.to_array()
    plt.title("Custom colors")
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
               interpolation="bilinear")
    new_filename = output_filename + ".png"
    wc.to_file(new_filename)


if __name__ == '__main__':
    ''' Generate wordclouds on all demo data '''

    # datascience cards
    data = 'data/datascience_flashcards.txt'
    df_datascience = clean.read_cards(data)
    df_datascience_clean = clean.clean_dataframe(df_datascience)
    df_datascience = clean.collapse_df(df_datascience_clean)

    wc.create_wordcloud_from_df(df_datascience, "/Users/tbot/Dropbox/galvanize/a-smarter-flashcard/images/brain_template.png")

    # biology cards
    data = 'data/biology_flashcards.txt'
    df_biology = clean.read_cards(data)
    df_biology_clean = clean.clean_dataframe(df_biology)
    df_biology = clean.collapse_df(df_biology_clean)

    wc.create_wordcloud_from_df(df_biology, "images/beaker_template.png")

    # history cards
    data = 'data/history_flashcards.txt'
    df_history = clean.read_cards(data)
    df_history_clean = clean.clean_dataframe(df_history)
    df_history = clean.collapse_df(df_history_clean)

    wc.create_wordcloud_from_df(df_history, "images/knight_template.png")
