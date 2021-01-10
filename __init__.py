from flask import Flask
from flask import request
from joblib import load
import numpy
import torch
import underthesea
from transformers import AutoModel, AutoTokenizer  # Thư viện BERT
from flask import jsonify


def create_app():
    app = Flask(__name__)
    max_len = 20
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    with open('save_model.pkl', 'rb') as fid:
        svm_model = load(fid)

    @app.route('/', methods=['GET', 'POST'])
    def hello():
        body = request.json
        text = body.get("sentence")
        print(text)
        v_tokenized = []
        # Phân thành từng từ
        line = underthesea.word_tokenize(text)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        # print("Word segment  = ", line)
        # Tokenize bởi BERT
        line = v_tokenizer.encode(line)
        v_tokenized.append(line)
        padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])

        # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
        attention_mask = numpy.where(padded == 1, 0, 1)

        # Chuyển thành tensor
        padded = torch.tensor(padded).to(torch.long)
        attention_mask = torch.tensor(attention_mask)

        # Lấy features dầu ra từ BERT
        with torch.no_grad():
            last_hidden_states = v_phobert(input_ids=padded, attention_mask=attention_mask)

        v_features = last_hidden_states[0][:, 0, :].numpy()
        sc = svm_model.predict_proba(v_features)
        cv = numpy.max(sc)  # Đây là giá trị probality
        cb = numpy.argmax(sc)  # Đây là giá trị index đạt max
        return jsonify(
            result=str(cb),
            cv=str(cv),
            text=text
        )

    return app
