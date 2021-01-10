import logging

import pandas as pd
from simpletransformers.ner import NERModel, NERArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    [0, "Tìm_kiếm", "O"],
    [0, "Sơn_Tùng", "B-PER"],
    [0, "trên", "O"],
    [0, "Youtube", "B-ORG"],
    [1, "Mở", "O"],
    [1, "Ronaldo", "B-PER"],
    [1, "trên", "O"],
    [1, "trình_duyệt", "B-ORG"],
    [2, "Tìm", "O"],
    [2, "thời_sự", "O"],
    [2, "trên", "O"],
    [2, "Youtube", "B-ORG"],
    [3, "Tìm", "O"],
    [3, "phim_hay", "O"],
    [3, "trên", "O"],
    [3, "Youtube", "B-ORG"],
    [4, "Tìm", "O"],
    [4, "bóng_đá", "O"],
    [4, "trên", "O"],
    [4, "Youtube", "B-ORG"],
]
train_data = pd.DataFrame(
    train_data, columns=["sentence_id", "words", "labels"]
)

eval_data = [
    [0, "Tìm", "O"],
    [0, "Xuân_Mai", "B-PER"],
    [0, "trên", "O"],
    [0, "Youtube", "B-ORG"],
    [1, "Tìm_kiếm", "O"],
    [1, "phim", "O"],
    [1, "vượt_ngục", "O"],
    [1, "Youtube", "B-ORG"],
]
eval_data = pd.DataFrame(
    eval_data, columns=["sentence_id", "words", "labels"]
)

# Configure the model
model_args = NERArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True

if __name__ == '__main__':
    model = NERModel(
        "roberta", "roberta-base", args=model_args, use_cuda=False)
    model.train_model(train_data, eval_data=eval_data)
    predictions, raw_outputs = model.predict(["Tìm bạn muốn hẹn hò trên Youtube"])
    print(predictions)
    print(raw_outputs)

# Train the model
# if __name__ == '__main__':
#     model.train_model(train_data, eval_data=eval_data)

# Evaluate the model
# result, model_outputs, preds_list = model.eval_model(eval_data)
#
# # Make predictions with the model
# predictions, raw_outputs = model.predict(["Hermione was the best in her class"])
