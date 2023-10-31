import os
import time
import torch
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

def load_models_from_checkpoints(folder_path):
    models = []
    for fold in os.listdir(folder_path):
        model_path = os.path.join(folder_path, fold)
        if os.path.isdir(model_path):
            model = BertForSequenceClassification.from_pretrained(model_path)
            models.append(model)
    return models


def soft_voting(models, tokenizer, device, texts):
    predictions = []

    for text in tqdm(texts, desc="Text Processing Progress"):
        predictions_for_text = []
        for model in models:
            model.to(device).eval()
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            predictions_for_text.append(probabilities)

        average_probabilities = np.mean(predictions_for_text, axis=0)

        predicted_label = np.argmax(average_probabilities, axis=1)
        predictions.append(predicted_label)

    return predictions

if __name__ == '__main__':
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    df = pd.read_excel('data/flitto_2023_data_team_test_input_2.xlsx')
    texts = df['text'].tolist()
    labels = df['lang_code'].tolist()

    trained_models = load_models_from_checkpoints('checkpoints')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    path_label2index = 'data/label_to_index.pkl'

    with open(path_label2index, 'rb') as f:
        label2index = pickle.load(f)

    index2label = {v:k for k,v in label2index.items()}

    df['detected'] = soft_voting(trained_models, tokenizer, device, df['text'])
    
    df['detected'] = df['detected'].apply(lambda x: index2label.get(x[0], x[0]))
    df.to_excel('output.xlsx', index=False)   

    df['match'] = df.apply(lambda row: row['lang_code'] == row['detected'], axis=1)

    match_count = df['match'].sum()
    total_count = len(df)
    match_ratio = match_count / total_count

    total_time = time.time() - start_time
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Match Ratio: {match_ratio * 100:.2f}%")
























# # 사전 훈련된 모델과 토크나이저 로드
# model_path = "trained_model/"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)

# # CUDA 사용 가능 여부 확인 및 모델 장치에 할당
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# model.eval()

# path_label2index = 'label_to_index.pkl'
# with open(path_label2index, 'rb') as f:
#     label2index = pickle.load(f)

# # 예측을 위한 함수
# def predict(text):
#     # 텍스트 토크나이징
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)
    
#     # 모델을 사용하여 예측
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#     logits = outputs.logits
#     predicted_label_idx = torch.argmax(logits, dim=1).item()

#     # 인덱스를 라벨로 변환
#     index2label = {v: k for k, v in label2index.items()}
#     return index2label[predicted_label_idx]

# # 예측 테스트
# text = "한국어 입니다."
# predicted_label = predict(text)
# print(f"Predicted Language: {predicted_label}")
