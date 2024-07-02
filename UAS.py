import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load data dari file CSV
df = pd.read_csv('scraped_data.csv')

# Konversi data ke tipe data string untuk memastikan kompatibilitas dengan TF-IDF Vectorizer
df['isi'] = df['isi'].astype(str)

# Fungsi pembersihan teks
def cleaning(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return text

df['Cleaning'] = df['isi'].apply(cleaning)

# Fungsi tokenisasi teks
def tokenizer(text):
    text = text.lower()
    return word_tokenize(text)

df['Tokenizing'] = df['Cleaning'].apply(tokenizer)

# Gabungkan kembali kata-kata setelah pembersihan dan penghapusan stopword
df['Final_Text'] = df['Tokenizing'].apply(lambda x: ' '.join(x))

# Hapus baris yang kosong setelah pembersihan
df = df[df['Final_Text'].str.strip().astype(bool)]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Final_Text']).toarray()
y = df['kategori']

# Pemisahan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling Catboost
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass', verbose=100)
model.fit(X_train, y_train)

# Fungsi untuk melakukan prediksi kategori berita dari teks input
def predict_category(text):
    text_cleaned = cleaning(text)
    text_tokenized = tokenizer(text_cleaned)
    text_final = ' '.join(text_tokenized)
    text_tfidf = vectorizer.transform([text_final])
    prediction = model.predict(text_tfidf)
    category = prediction[0]
    return category

# Streamlit app
st.title("Prediksi Kategori Berita")
st.write("Nama Kelompok:")
st.write("1. Septian Dio Dwinata.H")
st.write("2. Moch Akbar Bagas Prakasa")
st.write("3. Garvin Taufiqulhakim Faa’iz")
st.write("4. Garvan Taufiqurrahman Fawwaz")

# Input box untuk prediksi teks
input_text = st.text_input("Masukkan teks berita untuk diprediksi kategorinya:")
if st.button("Prediksi"):
    predicted_category = predict_category(input_text)
    st.write(f"Kategori prediksi untuk teks input adalah: {predicted_category}")

# Tampilkan seluruh dataset
if st.checkbox("Tampilkan dataset lengkap"):
    st.write(df)

# Tampilkan jumlah kategori
st.write("Jumlah Kategori :")
category_counts = df['kategori'].value_counts()
st.bar_chart(category_counts)

# Tampilkan hasil akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Akurasi Model: {accuracy:.4f}")

# Tampilkan Confusion Matrix
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
st.pyplot(fig)
