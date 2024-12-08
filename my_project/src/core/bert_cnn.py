import yaml
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AlbertModel
import torchvision.models as models
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Function to dynamically load the model based on the model type from YAML
def get_model(model_type, num_classes, rnn_hidden_size=None):
    if model_type == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)  # Replace final layer
    elif model_type == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
    elif model_type == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased')
        return BertForClassification(model, rnn_hidden_size, num_classes)
    elif model_type == 'albert':
        model = AlbertModel.from_pretrained('albert-base-v2')
        return BertForClassification(model, rnn_hidden_size, num_classes)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

    return model


# Custom model for BERT/ALBERT with optional RNN layer
class BertForClassification(nn.Module):
    def __init__(self, base_model, rnn_hidden_size, num_classes):
        super(BertForClassification, self).__init__()
        self.base_model = base_model
        self.rnn_hidden_size = rnn_hidden_size

        if rnn_hidden_size:  # If using an RNN layer
            self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                                hidden_size=rnn_hidden_size,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)
            self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # Bidirectional doubles hidden size
        else:
            self.fc = nn.Linear(self.base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        if self.rnn_hidden_size:
            lstm_output, _ = self.lstm(last_hidden_state)  # (batch_size, sequence_length, hidden_size*2)
            lstm_output = lstm_output[:, -1, :]  # Get last hidden state
            logits = self.fc(lstm_output)  # (batch_size, num_classes)
        else:
            pooled_output = last_hidden_state[:, 0, :]  # Take the CLS token (first token)
            logits = self.fc(pooled_output)  # (batch_size, num_classes)

        return logits


# Custom dataset class to prepare data for PyTorch
class SurveyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_len=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer:
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'image': text,  # For AlexNet/ResNet, this will be an image (replace text processing)
                'label': torch.tensor(label, dtype=torch.long)
            }


def main():
    # Load configuration
    config = load_config('/Users/mohsen/PycharmProjects/my_project/config/bert_rnn_rexnet__config.yml')

    # Load dataset
    df = pd.read_csv(config['dataset']['train_csv_path'])
    texts = df[config['dataset']['text_column']].tolist()
    labels = df[config['dataset']['label_column']].tolist()

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config['training']['train_val_split'], random_state=config['training']['random_state']
    )

    # Choose model type from YAML
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    rnn_hidden_size = config['model'].get('rnn_hidden_size', None)  # Only used for BERT/ALBERT

    # Load model
    model = get_model(model_type, num_classes, rnn_hidden_size)

    # For BERT/ALBERT, load tokenizer and process texts
    if model_type in ['bert', 'albert']:
        tokenizer = BertTokenizer.from_pretrained(config['tokenizer']['pretrained_model_name'])
        train_dataset = SurveyDataset(train_texts, train_labels, tokenizer, config['tokenizer']['max_length'])
        val_dataset = SurveyDataset(val_texts, val_labels, tokenizer, config['tokenizer']['max_length'])
    else:
        # If using AlexNet/ResNet, process data as images (you'll need image paths and preprocessing here)
        train_dataset = SurveyDataset(train_texts, train_labels)
        val_dataset = SurveyDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['validation']['batch_size'])

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if model_type in ['bert', 'albert']:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # For AlexNet/ResNet, assume 'image' is processed appropriately
                images = batch['image']
                labels = batch['label']
                logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}, Loss: {total_loss / len(train_loader)}')


if __name__ == "__main__":
    main()
