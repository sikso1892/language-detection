import os
import torch
import pickle

import requests
import zipfile

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding.input_ids.flatten(),
            'attention_mask': encoding.attention_mask.flatten(),
            'labels': torch.tensor(self.labels[idx])
        }

def train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs=3):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # 정확도 계산
            predicted_labels = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.shape[0]

        average_loss = total_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1)

            correct_predictions += (predicted_labels == true_labels).sum().item()
            total_predictions += true_labels.shape[0]

    accuracy = correct_predictions / total_predictions
    return accuracy

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_excel('data/flitto_2023_data_team_test_input_2.xlsx')
    texts = df['text']
    raw_labels = df['lang_code']

    path_label2index = 'data/label_to_index.pkl'

    if not os.path.exists(path_label2index):
        unique_labels = list(set(raw_labels))
        label2index = {label: index for index, label in enumerate(unique_labels)}
        with open(path_label2index, 'wb') as f:
            pickle.dump(label2index, f)

    else:
        with open(path_label2index, 'rb') as f:
            label2index = pickle.load(f)


    labels = [label2index[label] for label in raw_labels]
   
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    trained_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f'Fold: {fold + 1}')

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
    
        train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=16)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))
        optimizer = AdamW(model.parameters(), lr=1e-5)
        total_steps = len(train_dataloader) * 3
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs=3)
        accuracy = evaluate_model(model, val_dataloader, device)
        print(f"Validation Accuracy for Fold {fold + 1}: {accuracy * 100:.2f}%")

        model_save_path = f"checkpoints/trained_model_fold_{fold + 1}/"
        model.save_pretrained(model_save_path)
        trained_models.append(model)
