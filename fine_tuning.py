import csv
import sys
import os


checkpoint = sys.argv[1]
dataset_name = sys.argv[2]
learning_rate = float(sys.argv[3])
batch_size = int(sys.argv[4])

device = 'cuda'

patience = 5
num_epochs = 2
best_score = None
counter = 0

print('Fine tuning. Model:', checkpoint, '. Dataset:', dataset_name,
      '. Learning rate:', learning_rate, '. Batch size:', batch_size)

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

# Set separator
from Preprocessing.utils import set_separator

separator = set_separator(checkpoint)

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

### Model
import torch
from Modeling.model_torch import *
loss_fn = nn.CrossEntropyLoss()

model = CustomModel(checkpoint=checkpoint, num_labels=num_labels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### Train
## Configurations
import torch
from Preprocessing.utils import accuracy_fn, train_step, validation_step

for epoch in range(1, num_epochs+1):
    print(f"Epoch: {epoch}\n---------")
    train_loss, train_acc = train_step(model=model,
        split_size=train_size,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    validation_loss, validation_acc, _, _, _ = validation_step(model=model,
        split_size=validation_size,
        data_loader=validation_dataloader,
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
        # Save model
        base_folder = os.getcwd()

        model_name = checkpoint + '_' + dataset_name + '_' + str(epoch) + '_' + str(learning_rate) + '.pth'
        model_path = os.path.join(base_folder, 'Models', model_name)
        torch.save(best_model, model_path)

        # Test model
        test_loss, test_acc, f1_micro, f1_macro, f1_weighted = validation_step(model=best_model, split_size=test_size, data_loader=test_dataloader,
                                              loss_fn=loss_fn,accuracy_fn=accuracy_fn, device=device)

        # Save results
        results = [checkpoint, dataset_name, batch_size, best_epoch, learning_rate,
                   best_train_acc, best_train_loss.item(), best_validation_acc,
                   best_validation_loss.item(), test_acc, test_loss.item(),
                   f1_micro, f1_macro, f1_weighted]

        print(f'Test loss: {test_loss:.5f}. Accuracy: {test_acc:.2f}%. F1-micro: {f1_micro:.5f}. F1-macro: {f1_macro:.5f}. F1-weighted: {f1_weighted:.5f}.')

        base_folder = os.getcwd()
        file_path = f"{base_folder}/results_finetuning.csv"

        with open(file_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerows([results])

            break