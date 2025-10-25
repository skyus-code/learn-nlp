from sklearn.feature_extraction.text import CountVectorizer

texts = ["saya suka AI", "saya belajar AI"]

cv = CountVectorizer()
bow = cv.fit_transform(texts)
print("BoW:\n", bow.toarray())
print("Vocabulary:", cv.get_feature_names_out()) 