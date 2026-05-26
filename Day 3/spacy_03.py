import re

text = "her fiancé's résumé is beautiful"

def remove_accents(text):
    accents = re.compile(u"[\u0300-\u036F]|é|è")
    text = accents.sub(u"e", text)
    return text

cleaned_text = remove_accents(text)
print(cleaned_text)

"""
the output of this is given below
>>>> her fiance's resume is beautiful
"""

"""
U+0300 → U+036F

These are mostly combining accents/diacritics, such as:

̀ (grave accent)
́ (acute accent)
̂ (circumflex)
̃ (tilde)
"""