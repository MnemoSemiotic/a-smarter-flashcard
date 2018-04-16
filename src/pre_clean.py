import re

def strip_html(raw_html):
    '''
    INPUT:   string, potentially with html

    RETURNS: string of the text with html removed
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext








''' ######################################################################## ''' 
if __name__ == '__main__':
    strp_test = ''' 'ighted sum between the input-values and the weight-values, can mathematically be determined with the scalar-product <w, x>. To produce the behaviour of 'firing' a signal (+1) we can use the signum function sgn(); it maps the output to +1 if the input is positive, and it maps the output to -1 if the input is negative.<br><br>Thus, this Perceptron can mathematically be modeled by the function y = sgn(b+ <w, x>). Here b is the bias, i.e. the default value when all feature values are zero.         <div><img src=""quizlet-3PQBxmo2q4m6L2IeBD4nWw_m.png"" /></div>         "' '''

    print(strip_html(strp_test))
