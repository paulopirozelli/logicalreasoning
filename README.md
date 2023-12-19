This is the repository for the paper "Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models".


### Fine-tuning Transformer Models
To fine-tune the various transformer models for the Hypothesis Classification task, utilize the script fine_tuning.py.

Usage:
```bat
python fine_tuning.py <model_name> <dataset> <learning_rate> <batch_size> <base_folder>
```

Example:
```bat
python fine_tuning.py bert-base-uncased FOLIO 1e-6 64 /logicalreasoning
```

This command saves results in results_finetuning.csv and stores fine-tuned models in the 'Models' folder. For 'deberta', models are saved in a subfolder labeled 'microsoft'.

The CSV file includes the following data:

```
checkpoint, dataset_name, batch_size, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```

### Probing Comparison
To conduct the probing experiment, execute:

Usage:
```bat
python fine_tuning.py <model_name> <dataset> <learning_rate> <batch_size> <base_folder>
```

Example:
```bat
python probing_comparison.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 1layer 1e-06 64 /workspace/logicFT
```

This generates results_probing.csv with the following data:

```
model_name, dataset_name, classifier_type, batch_size, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```

### Layerwise Probing
For the layerwise probing experiment, run:

Usage:
```bat
python probing_layer.py <fine-tuned_model> <dataset> <classifier_type> <layer> <learning_rate> <batch_size> <base_folder>
```

Example:
```bat
python probing_comparison.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 1layer 1e-06 64 /workspace/logicFT
```

This saves results in results_probing.csv containing the following data:

```
model_name, dataset_name, classifier_type, batch_size, layer, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```
