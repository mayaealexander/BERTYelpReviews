import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import math 

# Read file
data = pd.read_csv('lotsofdata.csv')

# Define the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Create a configuration without loading the unused weights
config = BertConfig.from_pretrained(model_name)
config.num_labels = 2  # Number of labels for your classification task
config.load_weight = False

# Initialize the model with the customized configuration
model = BertForSequenceClassification.from_pretrained(model_name, config=config)

# Tokenize the reviews and convert them to tensors
reviews = data['Review'].tolist()
labels = data['Label'].apply(lambda x: 1 if x == 1 else 0).tolist()


# Tokenize and pad sequences using the tokenizer
encoded_data = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt', max_length=128)

input_ids = encoded_data['input_ids']
attention_mask = encoded_data['attention_mask']
labels = torch.tensor(labels)

# Combine input data into a TensorDataset
dataset = TensorDataset(input_ids, attention_mask, labels)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing sets with automatic padding
train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=SequentialSampler(train_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=8, sampler=SequentialSampler(test_dataset))

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


# Training loop with gradient accumulation
num_epochs = 15
accumulation_steps = 4
for epoch in range(num_epochs):
  model.train()
  total_loss = 0
  for step, batch in enumerate(train_dataloader):
      inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
      outputs = model(**inputs)
      loss = outputs.loss
      total_loss += loss.item()

      loss.backward()

      # Perform optimizer step after accumulation_steps batches
      if (step + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()

  # Adjust learning rate
  scheduler.step()

  # Validation
  model.eval()
  all_predictions = []
  all_labels = []
  with torch.no_grad():
      total_correct = 0
      total_samples = 0
      for batch in test_dataloader:
          inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
          outputs = model(**inputs)
          logits = outputs.logits
          predictions = torch.argmax(logits, dim=1)

          total_correct += (predictions == batch[2]).sum().item()
          total_samples += batch[2].size(0)

          all_predictions.extend(predictions.cpu().numpy())
          all_labels.extend(batch[2].cpu().numpy())

  # print confusion matrix
  cm = confusion_matrix(all_labels, all_predictions)
  print(f'Epoch {epoch + 1}/{num_epochs}, Confusion Matrix:')
  print(cm)

  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]

  accuracy = total_correct / total_samples
  print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}, Test Accuracy: {accuracy * 100:.2f}%')