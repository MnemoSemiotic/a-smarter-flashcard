import re
import pandas as pd
import replace_dictionary as replace
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string

def read_cards(file_path):
    '''
    INPUT: file_path with string of file holding flash card data

    OUTPUT: dataframe representing values read from file
    '''
    df = pd.read_csv(file_path, sep='\t', names=['question','answer'])
    return df

def clean_dataframe(df):
    df2 = df.copy()

    df2 = df2.applymap(lambda x: stopwords_stemmer(replace_with_dict(strip_html(x.lower()))) if type(x) is str else ' ')

    return df2

def strip_html(raw_html):
    '''
    INPUT:   string, potentially with html

    RETURNS: string of the text with html removed
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)

    return cleantext

def replace_with_dict(text):
    latex_remove = replace.get_replace_dict()
    latex_keys = latex_remove.keys()
    output = text

    for string in latex_remove:
        output = output.replace(string, ' ')

    return output

def collapse_df(df):
    df_collapsed = df.copy()

    df_collapsed["record"] = df["question"].map(str) + ' ' + df["answer"].map(str)
    return df_collapsed['record']

def stopwords_stemmer(text):
    text = re.sub(r'[^\w\s]','', text, re.UNICODE)
    stop = set(stopwords.words('english'))
    snowball = SnowballStemmer('english')
    text_split = text.split()
    text_snowball = [snowball.stem(word) for word in text_split if word not in stop]
    return_text = ' '.join(text_snowball)
    return return_text


if __name__ == '__main__':

    data = 'data/biology_flashcards.txt'
    df = read_cards(data)

    df_clean = clean_dataframe(df)

    # df_clean.tail()

    df_collapsed = collapse_df(df_clean)

    # df_collapsed.str.contains('ttt').nunique()
    df_collapsed[79]

    df_collapsed.isnull().sum()
    ## There are 110 NaN values after cleaning

    # Create series as mask for nan values
    nulls = pd.isnull(df_collapsed)

    df_collapsed[79]
    df['question'][79]
