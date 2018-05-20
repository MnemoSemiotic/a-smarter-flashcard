import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import path
import random
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

def create_wordcloud_from_df(df, filename=None):
    '''
    INPUT: pandas.core.series.Series

    OUTPUT: returns NONE
          : Calls create_wordcloud and generates text from
            pandas series
    '''

    list_of_strings = [str(i) for i in df]

    if filename==None:
        create_wordcloud(' '.join(list_of_strings))
    else:
        create_wordcloud_custom(' '.join(list_of_strings), filename)

def create_wordcloud(text):
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
    plt.savefig('wordmap_temp.png', dpi=dpi)


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def create_wordcloud_custom(text, image_filename):
    """
    Using custom colors
    ===================
    Using the recolor method and custom coloring functions.
    """

    d = path.dirname(__file__)

    # read image from filename
    mask = np.array(Image.open(path.join(d, image_filename)))

    wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, margin=1, random_state=1, collocations=False).generate(text)
    # store default colored image
    default_colors = wc.to_array()
    plt.title("Custom colors")
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
               interpolation="bilinear")
    new_filename = image_filename[:-4] + "_generated.png"
    wc.to_file(new_filename)


if __name__ == '__main__':
    pass
