from gensim.models import Word2Vec
import nltk

sentences = [
    "I love natural language processing",
    "I love machine learning",
    "Natural language processing is fun"
    "Machine learning is part of artificial intelligence"
]

tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
print("Tokenized Sentences:", tokenized_sentences)

# Latih Model Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100, # Dimensi vektor
    window=3,        # konteks kata di kiri-kanan
    min_count=1,     # kata muncul minimal 1x biar ikut dilatih
    sg=1             # 1 = skip-gram, 0 = CBOW
)

# Simpan model biar gak dilatih ulang nanti
model.save("word2vec_model.model")

print("Model Word2Vec berhasil dilatih dan disimpan!")
