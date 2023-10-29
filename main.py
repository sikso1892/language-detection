import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer, Linear, CrossEntropyLoss, LogSoftmax
from torch.optim import Adam

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from torchtext.vocab import Vocab

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

import requests
import zipfile
import os

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer, Linear, CrossEntropyLoss, LogSoftmax
from torch.optim import Adam
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import KFold

NUM_EPOCHS = 10
NUM_CHUNKS = 10000
BATCH_SIZE = 32
PATIENCE_LIMIT = 3
K_FOLDS = 5
LEARNING_RATE = 0.001

MODEL_PATH = 'language_model.pth'

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 정의
class LanguageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 모델 정의
class LanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_languages):
        super(LanguageModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        
        encoder_layers = TransformerEncoderLayer(embed_size, 8, )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)
        
        self.fc = Linear(embed_size, num_languages)
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # average pooling across the sequence
        x = self.fc(x)
        return self.log_softmax(x)

# 데이터 다운로드
def download_and_extract_data(url, download_path, filename=None):

    # Determine filename
    if filename is None:
        filename = url.split("/")[-1]

    filepath = os.path.join(download_path, filename)

    # Create download directory if it doesn't exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Download the file if it doesn't exist
    if not os.path.exists(filepath):
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} to {download_path} successfully.")
    else:
        print(f"{filename} already exists in {download_path}. Skipping download.")

    # Extract the contents
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(f"{download_path}/wili-2018")
        print(f"Extracted contents to {download_path}.")

# 데이터 토큰화
def data_process(raw_text_iter, vocab):
    data = [torch.tensor([vocab[token] for token in word_tokenize(item.lower())], dtype=torch.long) for item in raw_text_iter]
    return pad_sequence(data, padding_value=vocab['<pad>'])

def divide_into_chunks(data, labels, num_chunks):
    chunk_size = len(data) // num_chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_chunks - 1 else len(data)
        yield data[start_idx:end_idx], labels[start_idx:end_idx]
        
def train_one_epoch(model, data, labels, optimizer, loss_function, device, num_chunks=NUM_CHUNKS):
    model.train()
    total_loss = 0.0
    
    for data_chunk, labels_chunk in divide_into_chunks(data, labels, num_chunks):
        data_loader = DataLoader(LanguageDataset(data_chunk, labels_chunk), batch_size=BATCH_SIZE, shuffle=True)
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (len(data_loader) * num_chunks)

def validate(model, data, labels, loss_function, device, num_chunks=NUM_CHUNKS):
    model.eval()
    total_loss = 0.0
    
    for data_chunk, labels_chunk in divide_into_chunks(data, labels, num_chunks):
        data_loader = DataLoader(LanguageDataset(data_chunk, labels_chunk), batch_size=BATCH_SIZE)
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

    return total_loss / (len(data_loader) * num_chunks)


        
def main():
    url = "https://zenodo.org/record/841984/files/wili-2018.zip"
    extract_path = "./data"

    download_and_extract_data(url, extract_path)

    # 데이터 로드
    data_path = './data/wili-2018/'

    df_text = pd.read_csv(data_path + 'x_train.txt', names=['text'], sep='\t', header=None)
    df_label = pd.read_csv(data_path + 'y_train.txt', names=['label'], sep='\t', header=None)

    # 누락된 값을 제거
    df = pd.concat([df_text, df_label], axis=1).dropna()

    # 토큰화 및 어휘 사전 구축
    tokens = []
    for text in df['text']:
        tokens.extend(word_tokenize(text.lower()))

    vocab = Vocab(Counter(tokens))

    # 데이터 및 레이블 준비
    X = data_process(df['text'].tolist(), vocab)
    X = X.transpose(0, 1)
    label_mapping = {label: i for i, label in enumerate(df['label'].unique())}
    y = torch.tensor([label_mapping[label] for label in df['label'].tolist()])

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=2023)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_data_fold = X[train_idx]
        val_data_fold = X[val_idx]
        train_labels_fold = y[train_idx]
        val_labels_fold = y[val_idx]
        
        model = LanguageModel(len(vocab), 256, len(label_mapping)).to(device)
        optimizer = Adam(model.parameters())
        loss_function = CrossEntropyLoss().to(device)
        
        for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
            train_loss = train_one_epoch(model, train_data_fold, train_labels_fold, optimizer, loss_function, device)
            val_loss = validate(model, val_data_fold, val_labels_fold, loss_function, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the best model for this fold
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
                patience_counter = 0
                
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE_LIMIT:
                    print(f"Early stopping on fold {fold + 1}")
                    break
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")  
if __name__ == "__main__":
    main()
