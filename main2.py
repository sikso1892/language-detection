import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

# 데이터 및 레이블 로딩
with open("data/wili-2018/x_train.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

with open("data/wili-2018/y_train.txt", "r", encoding="utf-8") as f:
    labels = f.readlines()

# 데이터와 레이블을 함께 100분의 1로 샘플링
# random_state = np.random.RandomState(seed=2023)
# sampled_indexes = np.random.choice(len(texts), size=len(texts) // 100, replace=False)

# texts = np.array(texts)
# labels = np.array(labels)

# texts = texts[sampled_indexes]
# labels = labels[sampled_indexes]

import random
# 데이터와 레이블을 함께 100분의 1로 샘플링
sampled_indexes = random.sample(range(len(texts)), len(texts) // 1000)
texts = [texts[i] for i in sampled_indexes]
labels = [labels[i] for i in sampled_indexes]

unique_labels = list(set(labels))
label_to_id = {label: id for id, label in enumerate(unique_labels)}

labels = [label_to_id[label] for label in labels]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

input_ids = torch.tensor(encodings['input_ids'])
attention_masks = torch.tensor(encodings['attention_mask'])
label_tensors = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, label_tensors)

# 데이터 콜레이터 정의
def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_models = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"Training fold {fold + 1}")

    train_dataset = torch.utils.data.Subset(dataset, train_ids)
    eval_dataset = torch.utils.data.Subset(dataset, val_ids)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=len(unique_labels))
    
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold}',
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_steps=1000,
        logging_dir=f'./logs_fold_{fold}',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn  # 콜레이터 추가
    )

    trainer.train()
    
    fold_models.append(model)

# 각 모델을 사용하여 앙상블 기반의 추론을 수행합니다.
def ensemble_predict(models, input_ids, attention_masks):
    all_predictions = []
    
    for model in models:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.append(predictions)
    
    all_predictions = torch.stack(all_predictions)
    final_predictions = torch.mode(all_predictions, dim=0).values
    
    return final_predictions

# 예시로 첫 번째 폴드의 검증 데이터를 사용하여 추론을 진행합니다.
input_ids, attention_masks, true_labels = eval_dataset[:]
predictions = ensemble_predict(fold_models, input_ids, attention_masks)
accuracy = (predictions == true_labels).float().mean().item()

print(f"Ensemble accuracy: {accuracy * 100:.2f}%")
