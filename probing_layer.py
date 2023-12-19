import csv
import sys
import os
import torch
import torch.nn as nn

model_name = sys.argv[1]
dataset_name = sys.argv[2]
classifier_type = sys.argv[3]
layer = int(sys.argv[4])
learning_rate = float(sys.argv[5])
batch_size = int(sys.argv[6])

device = 'cuda'

patience = 5
num_epochs = 50
best_score = None
counter = 0

print('Probing Layer. Model:', model_name, '. Dataset:', dataset_name,
      '. Classifier:', classifier_type, '. Layer:', layer,
      '. Learning rate:', learning_rate, '. Batch size:', batch_size)

# For reproduction, set seed
torch.manual_seed(1234)

# Get model type
from Preprocessing.utils import check_model_type

base_model = check_model_type(model_name) # pretrained or finetuning
checkpoint = model_name.split('_')[0] # e.g., bert-base-uncased

# Set separator
from Preprocessing.utils import set_separator

separator = set_separator(checkpoint)

### Preprocess dataset
# Load dataset
if dataset_name == 'FOLIO':
    from Preprocessing.FOLIO import loading
elif dataset_name == 'LogicNLI':
    from Preprocessing.LogicNLI import loading
elif dataset_name == 'MultiNLI':
    from Preprocessing.MultiNLI import loading
elif dataset_name == 'RuleTaker':
    from Preprocessing.RuleTaker import loading
else:
    raise Exception('Invalid dataset')

df_train, df_validation, df_test = loading(separator)

# Number of labels
num_labels = len(df_train['label'].unique())

# Get dataset splits' sizes
train_size, validation_size, test_size = len(df_train), len(df_validation), len(df_test)

# Create DatasetDict
from Preprocessing.utils import dataset_dict

dataset_dict = dataset_dict(df_train, df_validation, df_test)

### Tokenization, padding, and transform into dataloader object
from Tokenization.token_padding import tokenizer_padding

train_dataloader, validation_dataloader, test_dataloader = tokenizer_padding(dataset_dict,
                                                                             checkpoint, batch_size)

# Model path
model_path = os.path.join('Models', model_name)

#### Probing
## Create classifier
# Get input size
from Preprocessing.utils import input_size

input_size = input_size(checkpoint)

# Create clasifier
if classifier_type == '1layer':
    head = nn.Linear(input_size, num_labels)
elif classifier_type == '3layers':
    head = nn.Sequential(nn.Dropout(0.5),
                         nn.Linear(input_size, 256),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(256, 64),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(64, num_labels))

class CustomModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = torch.load(model_path)
        self.model.layer1 = nn.Identity()
        self.model.layer2 = nn.Identity()
        self.model.classifier = nn.Identity()

        self.sequential = head

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        x = outputs[0][1][layer][:,0,:]
        logits = self.sequential(x)

        return logits

model = CustomModel(num_labels=num_labels).to(device)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Freeze body of the model
# Turn off requires_grad for all layers
for param in model.parameters():
    param.requires_grad = False

# Turn on requires_grad for classifier parameters
for parameter in model.sequential.parameters():
   parameter.requires_grad = True

### Train
import torch
from Preprocessing.utils import accuracy_fn, train_layer, eval_layer

#Implement patience
for epoch in range(1, num_epochs+1):
    print(f"Epoch: {epoch}\n---------")
    train_loss, train_acc = train_layer(data_loader=train_dataloader,
        split_size=train_size,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    validation_loss, validation_acc, _, _, _ = eval_layer(data_loader=validation_dataloader,
        split_size = validation_size,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

    print(f"Validation loss: {validation_loss:.5f} | Validation accuracy: {validation_acc:.2f}%")

    # For the first epoch
    if best_score is None:
        best_train_acc = train_acc
        best_train_loss = train_loss
        best_validation_acc = validation_acc
        best_validation_loss = validation_loss
        best_epoch = epoch
        best_model = model  # Copy best model
    # For the other epochs
    else:
        # Check if the validation loss has decreased. If true, update the best results
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_validation_acc = validation_acc
            best_epoch = epoch
            best_model = model  # Copy best model
        else:
        # If validation loss has not decreased, add 1 to the counter
            counter = counter + 1

    if counter == patience or epoch == num_epochs:
        # Test and save results
        test_loss, test_acc, f1_micro, f1_macro, f1_weighted = eval_layer(model=best_model, split_size=test_size, data_loader=test_dataloader, loss_fn=loss_fn,
                                   accuracy_fn=accuracy_fn, device=device)

        results = [model_name, dataset_name, classifier_type, batch_size, layer, best_epoch,
                   learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc,
                   best_validation_loss.item(), test_acc, test_loss.item(),
                   f1_micro, f1_macro, f1_weighted]

        print(f'Test loss: {test_loss:.5f}. Accuracy: {test_acc:.2f}%. F1-micro: {f1_micro:.5f}. F1-macro: {f1_macro:.5f}. F1-weighted: {f1_weighted:.5f}.')

        base_folder = os.getcwd()
        file_path = f"{base_folder}/results_layer.csv"

        with open(file_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerows([results])

            break

