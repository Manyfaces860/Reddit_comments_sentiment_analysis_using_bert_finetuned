import torch
import torch.nn as nn
import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score


class YayOrNay(nn.Module):
    def __init__(self, output_size):
        super(YayOrNay, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


class Custom(Dataset):
    def __init__(self, data_path, transform=None, start=0, limit=2000):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        text = self.data.iloc[index, 1]
        target = self.data.iloc[index, 2]
        if self.transform:
            text = self.transform(text)
        text['label'] = torch.tensor(target, dtype=torch.long)
        return text


def evaluate_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    predictions, true_labels, val_loss = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())

            logits = outputs
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1, sum(val_loss)/len(val_loss), predictions


def make_pred(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('mps')

    model = torch.load("/path-to-model-dir/emotion_iden.pth", map_location=device, weights_only=False)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(device)

    def transformer(x):
        return tokenizer(x, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    with open("predict.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["", "text", "label"])   
        writer.writerow([0, f"""{text}""", 0])

    test_dataset = Custom('predict.csv', transformer)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    val_accuracy, f1, val_loss, preds = evaluate_model(model, testloader, device)
    return preds[0]


# print(make_pred("i am just good"))
