from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
text = "He is studying hard"
words = word_tokenize(text)

lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
print("Lemmatized words:", lemmatized)