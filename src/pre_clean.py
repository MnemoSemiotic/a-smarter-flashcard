import re
import pandas as pd


def read_cards(file_path):
    '''
    INPUT: file_path with string of file holding flash card data

    OUTPUT: dataframe representing values read from file
    '''
    df = pd.read_csv(file_path, sep='\t', names=['question','answer'])
    return df

def clean_dataframe(df):
    df2 = df.copy()

    df2 = df2.applymap(lambda x: strip_latex(strip_html(x)) if type(x) is str else x)


    # Remove html
    # df2.loc[:,["question", "answer"]] = df2.loc[:,["question", "answer"]].apply(lambda x: strip_html(str(x)))
    # df2['question'] = df['question'].map(lambda x: strip_latex(x))
    # df2['answer']   = df['answer'].map(lambda x: strip_html(str(x)))
    # df2['answer']   = df['answer'].map(lambda x: strip_latex(str(x)))

    return df2

def strip_html(raw_html):
    '''
    INPUT:   string, potentially with html

    RETURNS: string of the text with html removed
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cl_1 = cleantext.replace('&nbsp;',' ')
    cl_2 = cl_1.replace('&gt;','>')
    cl_3 = cl_2.replace('&lt;','<')
    return cl_3

def strip_latex(text):
    cl_1 = text.replace('\item', ' ')
    return cl_1

def collapse_df(df):
    df_collapsed = df.copy()
    df_collapsed['record'] = df['question'] + ' ' + df['answer']
    return df_collapsed['record']

''' ######################################################################## '''
if __name__ == '__main__':
    strp_test = '''What three summary statistics do all four of these scatterplots have in common? <br /><img src="Screen shot 2012-06-26 at 10.00.02 PM.png" /> mean, standard deviation, and Pearson correlation (http://en.wikipedia.org/wiki/Anscombe\'s_quartet) <div>&gt; the moral is that these summary statistics can be misleading and it\'s good to look at the actual distribution&nbsp;</div>'''
    print(strip_html(strp_test))


    data = 'data/ds_flashcards_2.txt'
    df = read_cards(data)
    df['question'][0]
    df_clean = clean_dataframe(df)

    df_clean['question'][0]
    
