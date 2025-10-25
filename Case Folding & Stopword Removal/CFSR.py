from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

text = "Saya ingin pergi ke pasar tapi hujaan deras sekali, jadi saya urungkan niat saya."

tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('indonesian'))

filtered_words = [word for word in tokens if word not in stop_words]

print(filtered_words)
