import src.pre_clean as clean
import pandas as pd


if __name__ == '__main__':
    data = 'data/ds_flashcards.txt'
    df = clean.read_cards(data)

    df_clean = clean.clean_dataframe(df)

    df_clean.tail()

    df.tail()
    
