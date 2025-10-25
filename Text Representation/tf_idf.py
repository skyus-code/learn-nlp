from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["saya suka AI", "saya belajar AI"]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts)
print("\nTF-IDF:\n", tfidf_matrix.toarray())
print("Vocabulary:", tfidf.get_feature_names_out())