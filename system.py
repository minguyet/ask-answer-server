from sklearn.feature_extraction.text import TfidfVectorizer
import underthesea

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Hàm chuẩn hoá câu
def standardize_data(row):

    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", "").replace(".", "").replace(";", "").replace("“", "").replace(":", "").replace("”", "").replace('"', "").replace("'", "").replace("!", "").replace("?", "").replace("-", "").replace("?", "")
    row = row.strip().lower()
    
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("stopword.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        
        sw.append(line.replace("\n",""))
    return sw


def get_data():
    v_text = []
    with open("dataset.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        v_text.append(line.replace("\n",""))     
    return v_text


def processing(lines):
    v_text = []
    for line in lines:
        v_text.append(standardize_data(line))
    return v_text


def token(v_text):
    global sw
    v_tokenized = []
    for i_text in v_text:
        line = underthesea.word_tokenize(i_text)
        filtered_sentence = [w for w in line if not w in sw]
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        v_tokenized.append(line)
    return v_tokenized


def load_data():
    global sw, data, tfidf_vectorizer    
    sw = load_stopwords()
    data = get_data()
    tfidf_vectorizer = TfidfVectorizer()

def get_answer(ans):
    documents = data.copy()
    documents.append(ans)
    documents = processing(documents)
    documents = token(documents)
    sparse_matrix = tfidf_vectorizer.fit_transform(documents)
    si = cosine_similarity(sparse_matrix, sparse_matrix)[len(documents)-1]
    result = zip(range(len(si)), si)
    result = sorted(result, key=lambda x: x[1], reverse= True)
    answer = []
    for x in range(1,6):
        answer.append(data[result[x][0]])    
    return answer



