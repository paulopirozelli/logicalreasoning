This is the repository for the paper "Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models".

### Abstract
Logical reasoning is central to complex human activities, such as thinking, debating, and planning; it is also a central component of many AI systems as well. In this paper, we investigate the extent to which encoder-only transformer language models (LMs) can reason according to logical rules. We ask whether those LMs can deduce theorems in propositional calculus and first-order logic; if their relative success in these problems reflects general logical capabilities; and which layers contribute the most to the task. First, we show for several encoder-only LMs that they can be trained, to a reasonable degree, to determine logical validity on various datasets. Next, by cross-probing fine-tuned models on these datasets, we show that LMs have difficulty in transferring their putative logical reasoning ability, which suggests that they may have learned dataset-specific features, instead of a general capability. Finally, we conduct a layerwise probing experiment, which shows that the hypothesis classification task is mostly solved through higher layers.

## Code

Bash commands should be run in the project directory.

### Fine-tuning Transformer Models
To fine-tune the various transformer models for the Hypothesis Classification task (sec. 4), utilize the script fine_tuning.py.

Usage:
```bat
python fine_tuning.py <model_name> <dataset> <learning_rate> <batch_size>
```

Example:
```bat
python fine_tuning.py bert-base-uncased FOLIO 1e-6 64
```

This command saves results in results_finetuning.csv and stores fine-tuned models in the 'Models' folder. For the logformer and deberta models, fine-tuned models are saved in subfolders labeled 'allenai' and 'microsoft', respectively.

The CSV file includes the following data:

```
checkpoint, dataset_name, batch_size, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```

The full list of enconder-only transformer models tested in this task are: 

```
distilbert-base-uncased, bert-base-uncased, bert-large-uncased, roberta-base, roberta-large, allenai/longformer-base-4096, allenai/longformer-large-4096, microsoft/deberta-v3-xsmall, microsoft/deberta-v3-small, microsoft/deberta-v3-base, microsoft/deberta-v3-large, microsoft/deberta-v2-xlarge, microsoft/deberta-v2-xxlarge, albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2, xlm-roberta-base, xlm-roberta-large, xlnet-base-cased, xlnet-large-cased
```

### Probing Comparison
To conduct the probing experiment (sec. 5), execute:

Usage:
```bat
python probing_comparison.py <model_name> <dataset> <classifier_type> <learning_rate> <batch_size>
```

Example:
```bat
python probing_comparison.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 1layer 1e-06 64
```

This generates results_probing.csv with the following data:

```
model_name, dataset_name, classifier_type, batch_size, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```

### Layerwise Probing
For the layerwise probing experiment (sec. 6), run:

Usage:
```bat
python probing_layer.py <fine-tuned_model> <dataset> <classifier_type> <layer> <learning_rate> <batch_size>
```

Example:
```bat
python probing_layer.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 1layer 12 1e-06 64
```

This saves results in results_probing.csv containing the following data:

```
model_name, dataset_name, classifier_type, batch_size, layer, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```
