import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import zipfile

# Function to download and extract the dataset
def download_and_extract_dataset(url, save_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    if not os.path.exists(save_path):
        print(f"Downloading MovieLens 20M Dataset from {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=save_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                f.write(data)
    
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"Dataset extracted to {extract_path}")

# Custom Dataset class
class MovieDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

# BERT4Rec model class
class BERT4Rec(torch.nn.Module):
    def __init__(self, config):
        super(BERT4Rec, self).__init__()
        self.bert = BertModel(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state)
        return logits

# Function to recommend movies
def recommend_movies(user_sequence, model, tokenizer, top_k=5):
    model.eval()
    inputs = tokenizer(user_sequence, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
    input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    logits = outputs[0, -1, :]
    top_k_indices = torch.topk(logits, top_k).indices.cpu().numpy()
    return [tokenizer.decode([idx]) for idx in top_k_indices]

# Main function
if __name__ == "__main__":
    # Download and extract the dataset
    root = "ml-20m"
    file_name = root + ".zip"
    dataset_url = "https://files.grouplens.org/datasets/movielens/" + file_name
    save_path = file_name
    extract_path = "."

    download_and_extract_dataset(dataset_url, save_path, extract_path)

    # Load the MovieLens dataset
    ratings = pd.read_csv(root + '/ratings.csv')
    movies = pd.read_csv(root + '/movies.csv')
    tags = pd.read_csv(root + '/tags.csv')

    # Preprocess the data
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings.sort_values(by=['userId', 'timestamp'])

    # Create sequences of movie interactions for each user
    user_sequences = ratings.groupby('userId')['movieId'].apply(list).reset_index()

    # Split the data into train, validation, and test sets
    train_sequences, test_sequences = train_test_split(user_sequences['movieId'].tolist(), test_size=0.2, random_state=42)
    train_sequences, val_sequences = train_test_split(train_sequences, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Initialize the tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = MovieDataset(train_sequences, tokenizer, max_len=100)
    val_dataset = MovieDataset(val_sequences, tokenizer, max_len=100)
    test_dataset = MovieDataset(test_sequences, tokenizer, max_len=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    config = BertConfig(vocab_size=len(movies) + 1, hidden_size=256, num_hidden_layers=6, num_attention_heads=8)
    model = BERT4Rec(config)
    model = model.to('cuda')

    # Training loop with early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 2
    epochs_no_improve = 0

    for epoch in range(50):  # Set a high number of epochs to allow for early stopping
        model.train()
        train_loss = 0
        for input_ids, attention_mask in train_loader:
            input_ids, attention_mask = input_ids.to('cuda'), attention_mask.to('cuda')
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, config.vocab_size), input_ids.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask in val_loader:
                input_ids, attention_mask = input_ids.to('cuda'), attention_mask.to('cuda')
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, config.vocab_size), input_ids.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Evaluation on the test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask in test_loader:
            input_ids, attention_mask = input_ids.to('cuda'), attention_mask.to('cuda')
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, config.vocab_size), input_ids.view(-1))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')

    # Example usage
    user_sequence = [1, 2, 3, 4, 5]  # Replace with actual movie IDs from a user's history
    recommended_movies = recommend_movies(user_sequence, model, tokenizer)
    print("Recommended Movies:", recommended_movies)
