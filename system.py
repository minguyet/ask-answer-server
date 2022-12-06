# Define the documents

#documents = [doc_trump, doc_election,doc_putin,  test]
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
    #print("stop word:", sw)
    return sw


def get_data():
    v_text = []

    data = []
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
    max_len = 100 # Mỗi câu dài tối đa 100 từ
    for i_text in v_text:
        #print("Đang xử lý line = ", i_text)
        # Phân thành từng từ
        line = underthesea.word_tokenize(i_text)
        # Lọc các từ vô nghĩa
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
    print("")
    documents = data.copy()
    documents.append(ans)
    #print(documents)
    documents = processing(documents)
    documents = token(documents)
    
    sparse_matrix = tfidf_vectorizer.fit_transform(documents)
    si = cosine_similarity(sparse_matrix, sparse_matrix)[len(documents)-1]
    result = zip(range(len(si)), si)
    result = sorted(result, key=lambda x: x[1], reverse= True)
    answer = data[result[1][0]]
    return answer



