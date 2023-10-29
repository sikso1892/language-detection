from sklearn.model_selection import KFold
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

# 초기 설정
K = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 및 레이블 로딩
with open("data/wili-2018/x_train.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

with open("data/wili-2018/y_train.txt", "r", encoding="utf-8") as f:
    labels = f.readlines()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 토크나이저 및 모델 초기화
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# 데이터 토큰화
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
input_ids, attention_masks = encoded_texts["input_ids"], encoded_texts["attention_mask"]

# K-fold 시작
kf = KFold(n_splits=K)
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(input_ids)):
    print(f"Training Fold {fold + 1}")

    train_inputs = input_ids[train_idx].to(device)
    val_inputs = input_ids[val_idx].to(device)
    train_labels = torch.tensor(encoded_labels[train_idx]).to(device)
    val_labels = torch.tensor(encoded_labels[val_idx]).to(device)
    train_masks = attention_masks[train_idx].to(device)
    val_masks = attention_masks[val_idx].to(device)

    # 모델 정의 및 GPU로 이동
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(label_encoder.classes_)).to(device)

    # Trainer 및 TrainingArguments 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=f'./logs/fold_{fold}',
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=500,
        save_steps=500,
        push_to_hub=False,
        output_dir=f'./results/fold_{fold}'
    )

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 훈련 시작
    trainer.train()
    
    # 모델 저장
    model_path = f'./model_fold_{fold}'
    model.save_pretrained(model_path)
    fold_models.append(model_path)

# 예측 시
def ensemble_predict(models, input_ids, attention_mask):
    all_predictions = []

    with torch.no_grad():
        for model_path in models:
            model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_predictions.append(logits)

    # 모든 모델의 결과를 합산
    avg_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    return torch.argmax(avg_predictions, dim=1)

# 예시 예측
input_text = ["Example text 1", "Example text 2"]
encoded = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=256)
input_ids, attention_mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
predictions = ensemble_predict(fold_models, input_ids, attention_mask)
print(predictions)
