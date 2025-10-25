import re
from nltk.tokenize import word_tokenize
import nltk

text = "This is a sample text to be normalized. It includes numbers like 123 and punctuation!"

text = text.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)

token = word_tokenize(text)

print('Tokenize : ',token)
print('Normalize : ', text)