import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy
import re
import underthesea

from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer

from sklearn.svm import SVC
from joblib import dump


def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer


def standardize_data(row):
    row = re.sub(r"[.,?]+$-", "", row)
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ")
    row = row.strip()
    return row


def load_stopwords():
    sw = []
    with open("remove.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n", ""))
    return sw


def load_data():
    v_text = []
    v_label = []

    with open('data_1.csv', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.replace("\n", "")
        print(line[:-2])
        v_text.append(standardize_data(line[:-2]))
        v_label.append(int(line[-1:].replace("\n", "")))
    print(v_label)
    return v_text, v_label


def make_bert_features(v_text):
    global phobert, sw
    v_tokenized = []
    max_len = 100
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        line = underthesea.word_tokenize(i_text)
        filtered_sentence = [w for w in line if not w in sw]
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        line = tokenizer.encode(line)
        v_tokenized.append(line)
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    print('padded:', padded[0])
    print('len padded:', padded.shape)

    attention_mask = numpy.where(padded == 1, 0, 1)

    # Chuyển thành tensor
    padded = torch.tensor(padded).to(torch.long)
    print("Padd = ", padded.size())
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = phobert(input_ids=padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    print(v_features.shape)
    return v_features


sw = load_stopwords()

phobert, tokenizer = load_bert()

text, label = load_data()
features = make_bert_features(text)

# Phân chia dữ liệu train, test
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.15, random_state=0)

#
# parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 4], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(SVC(), param_grid=parameters)
# grid_search = clf.fit(X_train, y_train)
#
# print("Best score: %0.3f" % grid_search.best_score_)
# print(grid_search.best_estimator_)
#
# # best prarams
# print('best prarams:', clf.best_params_)

print("Chuẩn bị train model SVM....")
cl = SVC(kernel='linear', probability=True, gamma=0.125)
cl.fit(features, label)

sc = cl.score(X_test, y_test)
print('Kết quả train model, độ chính xác = ', sc * 100, '%')

dump(cl, 'save_model.pkl')
print("Đã lưu model SVM vào file save_model.pkl")
