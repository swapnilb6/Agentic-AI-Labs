#Punctuation Removal

import re

text = ' (to love is to destroy, and to be loved, is to be "the" one <destroyed>} '

def remove_punctuations(text):
    punctuation = re.compile(r'[{};():,."/<>-]')
    text = punctuation.sub(' ', text)
    return text

clean_text = remove_punctuations(text)
print(clean_text)

"""
the output of this is given below : 
>>>> to love is to destroy  and to be loved  is to be  the  one  destroyed
"""