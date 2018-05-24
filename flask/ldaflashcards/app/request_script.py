


if __name__ == '__main__':
    p = preparedata()
    cleaned_df = p.apply()
    path = '../models/finalized_model.pkl'
    # with open(path) as f:
    model = pickle.load(open('../models/finalized_model.pkl', 'rb'))
    print(cleaned_df.columns)
    # pred1 = str(model.predict_proba([data])[0])
    pred = p.predict_proba(model, cleaned_df)

    print('Likelihood that this is fraud: {0:0.2%}'.format(pred[0][1]))
    # new_df = cleaned_df.apply_functions(LDA=False)
