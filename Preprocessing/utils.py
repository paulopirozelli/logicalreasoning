import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import evaluate
torch.manual_seed(1234)

def dataset_dict(df_train, df_validation, df_test):
    train_dataset = Dataset.from_dict(df_train)
    validation_dataset = Dataset.from_dict(df_validation)
    test_dataset = Dataset.from_dict(df_test)

    return datasets.DatasetDict({"train":train_dataset, 'validation': validation_dataset,
                                 "test":test_dataset})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc



# train and test steps
def train_step(model, split_size, data_loader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    total_loss = 0

    predictions = []
    correct_labels = []

    for batch in data_loader:
        # Send data to GPU
        # Send data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # 1. Forward pass
        output, _, _ = model(**batch)

        # 2. Calculate loss
        loss = loss_fn(output.logits, batch['labels'])
        total_loss += loss * batch['labels'].size(0)  # for each cuda processing, multiple loss by the number of examples

        # Get results
        y_pred = torch.argmax(output.logits, dim=1)  # get predicted class in each process
        y_label = batch['labels']

        predictions.append(y_pred)
        correct_labels.append(y_label)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Concatenate tensors
    predictions = torch.cat(predictions, dim=0)
    correct_labels = torch.cat(correct_labels, dim=0)

    # Calculate loss and accuracy per epoch and print out what's happening
    mean_loss = total_loss / split_size
    mean_acc = accuracy_fn(predictions, correct_labels)

    return mean_loss, mean_acc


def validation_step(model, split_size, data_loader, loss_fn, accuracy_fn, device):
    total_loss = 0
    model.eval()  # put model in eval mode

    predictions = []
    correct_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        for batch in data_loader:
            # Send data to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1. Forward pass
            output, _, _ = model(**batch)

            # 2. Calculate loss
            loss = loss_fn(output.logits, batch['labels'])
            total_loss += loss * batch['labels'].size(0)  # for each cuda processing, multiple loss by the number of examples

            # Get results
            y_pred = torch.argmax(output.logits, dim=1)  # get predicted class in each process
            y_label = batch['labels']

            predictions.append(y_pred)
            correct_labels.append(y_label)

        # Concatenate tensors
        predictions = torch.cat(predictions, dim=0)
        correct_labels = torch.cat(correct_labels, dim=0)

        # Calculate loss and accuracy per epoch and print out what's happening
        mean_loss = total_loss / split_size
        mean_acc = accuracy_fn(predictions, correct_labels)

        # Calculate F1-score
        f1_metric = evaluate.load("f1")
        f1_micro = f1_metric.compute(predictions=predictions, references=correct_labels, average="micro")['f1']
        f1_macro = f1_metric.compute(predictions=predictions, references=correct_labels, average="macro")['f1']
        f1_weighted = f1_metric.compute(predictions=predictions, references=correct_labels, average="weighted")['f1']

        return mean_loss, mean_acc, f1_micro, f1_macro, f1_weighted


## Train probing
def train_probing(data_loader, split_size, model, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    total_loss = 0

    predictions = []
    correct_labels = []

    for batch in data_loader:
        # Send data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # 1. Forward pass
        output = model(**batch)

        # 2. Calculate loss
        loss = loss_fn(output, batch['labels'])
        total_loss += loss * batch['labels'].size(0)  # for each cuda processing, multiple loss by the number of examples

        # Get results
        y_pred = torch.argmax(output, dim=1)  # get predicted class in each process
        y_label = batch['labels']

        predictions.append(y_pred)
        correct_labels.append(y_label)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()


    # Concatenate tensors
    predictions = torch.cat(predictions, dim=0)
    correct_labels = torch.cat(correct_labels, dim=0)

    # Calculate loss and accuracy per epoch and print out what's happening
    mean_loss = total_loss / split_size
    mean_acc = accuracy_fn(predictions, correct_labels)

    return mean_loss, mean_acc


def eval_probing(data_loader, model, loss_fn, accuracy_fn, split_size, device):
    total_loss = 0
    model.eval()  # put model in eval mode

    predictions = []
    correct_labels = []

    with torch.inference_mode():
        for batch in data_loader:
            # Send data to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1. Forward pass
            output = model(**batch)

            # 2. Calculate loss
            loss = loss_fn(output, batch['labels'])
            total_loss += loss * batch['labels'].size(0)  # for each cuda processing, multiple loss by the number of examples

            # Get results
            y_pred = torch.argmax(output, dim=1)  # get predicted class in each process
            y_label = batch['labels']

            predictions.append(y_pred)
            correct_labels.append(y_label)

        # Concatenate tensors
        predictions = torch.cat(predictions, dim=0)
        correct_labels = torch.cat(correct_labels, dim=0)

        # Calculate loss and accuracy per epoch and print out what's happening
        mean_loss = total_loss / split_size
        mean_acc = accuracy_fn(predictions, correct_labels)

        # Calculate F1-score
        f1_metric = evaluate.load("f1")
        f1_micro = f1_metric.compute(predictions=predictions, references=correct_labels, average="micro")['f1']
        f1_macro = f1_metric.compute(predictions=predictions, references=correct_labels, average="macro")['f1']
        f1_weighted = f1_metric.compute(predictions=predictions, references=correct_labels, average="weighted")['f1']

        return mean_loss, mean_acc, f1_micro, f1_macro, f1_weighted

def train_layer(data_loader, split_size, model, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    total_loss = 0

    predictions = []
    correct_labels = []

    for batch in data_loader:
        # Send data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # 1. Forward pass
        output = model(**batch)

        # 2. Calculate loss
        loss = loss_fn(output, batch['labels'])
        total_loss += loss * batch['labels'].size(0)  # for each cuda processing, multiple loss by the number of examples

        # Get results
        y_pred = torch.argmax(output, dim=1)  # get predicted class in each process
        y_label = batch['labels']

        predictions.append(y_pred)
        correct_labels.append(y_label)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()


    # Concatenate tensors
    predictions = torch.cat(predictions, dim=0)
    correct_labels = torch.cat(correct_labels, dim=0)

    # Calculate loss and accuracy per epoch and print out what's happening
    mean_loss = total_loss / split_size
    mean_acc = accuracy_fn(predictions, correct_labels)

    return mean_loss, mean_acc

def eval_layer(data_loader, model, loss_fn, accuracy_fn, split_size, device):
    total_loss = 0
    model.eval()  # put model in eval mode

    predictions = []
    correct_labels = []

    with torch.inference_mode():
        for batch in data_loader:
            # Send data to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1. Forward pass
            output = model(**batch)

            # 2. Calculate loss
            loss = loss_fn(output, batch['labels'])
            total_loss += loss * batch['labels'].size(0)  # for each cuda processing, multiple loss by the number of examples

            # Get results
            y_pred = torch.argmax(output, dim=1)  # get predicted class in each process
            y_label = batch['labels']

            predictions.append(y_pred)
            correct_labels.append(y_label)

        # Concatenate tensors
        predictions = torch.cat(predictions, dim=0)
        correct_labels = torch.cat(correct_labels, dim=0)

        # Calculate loss and accuracy per epoch and print out what's happening
        mean_loss = total_loss / split_size
        mean_acc = accuracy_fn(predictions, correct_labels)

        # Calculate F1-score
        f1_metric = evaluate.load("f1")
        f1_micro = f1_metric.compute(predictions=predictions, references=correct_labels, average="micro")['f1']
        f1_macro = f1_metric.compute(predictions=predictions, references=correct_labels, average="macro")['f1']
        f1_weighted = f1_metric.compute(predictions=predictions, references=correct_labels, average="weighted")['f1']

        return mean_loss, mean_acc, f1_micro, f1_macro, f1_weighted


def get_num_layers(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)
    return config.num_hidden_layers

def check_model_type(model_name):
  try:
    model_name.split('_')[1]
  except IndexError:
    return 'pretrained'
  else:
    return 'finetuning'



def input_size(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)
    return config.hidden_size

# Set separator
def set_separator(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Get the special tokens from the tokenizer
    separator = tokenizer.special_tokens_map['sep_token']

    return separator

