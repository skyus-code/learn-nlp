from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
text = "He running on the park"
words = word_tokenize(text)

stemmed = [stemmer.stem(word) for word in words]
print("Stemmed words:", stemmed)