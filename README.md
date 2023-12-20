This is the repository for the paper ["Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models"](https://arxiv.org/abs/2312.11720).

> **Abstract:** Logical reasoning is central to complex human activities, such as thinking, debating, and planning; it is also a central component of many AI systems as well. In this paper, we investigate the extent to which encoder-only transformer language models (LMs) can reason according to logical rules. We ask whether those LMs can deduce theorems in propositional calculus and first-order logic; if their relative success in these problems reflects general logical capabilities; and which layers contribute the most to the task. First, we show for several encoder-only LMs that they can be trained, to a reasonable degree, to determine logical validity on various datasets. Next, by cross-probing fine-tuned models on these datasets, we show that LMs have difficulty in transferring their putative logical reasoning ability, which suggests that they may have learned dataset-specific features, instead of a general capability. Finally, we conduct a layerwise probing experiment, which shows that the hypothesis classification task is mostly solved through higher layers.

## Results 
Full results for the experiments are found in the sheet [results.xlsx](https://github.com/paulopirozelli/logicalreasoning/blob/main/results.xlsx). 

## Datasets
To facilitate reproduction, we organized all datasets used in this paper in a single folder, which can be found in our Google Drive [folder](https://drive.google.com/drive/folders/1YpRoveEJJZIOUyAMeeo5LF6kt8eAFkya) (5.18GB). Download this folder and replace the original 'LogicData' folder from the repository.

Appendix A of the paper describes the datasets in detail and indicates the sources they were extracted from.

## Code
Bash commands should be run in the main directory.

The dataset names are:

```
FOLIO, LogicNLI, RuleTaker, SimpleLogic
```

### Fine-tuning Transformer Models
To fine-tune the various transformer models for the Hypothesis Classification task (sec. 4), utilize the script fine_tuning.py.

Usage:
```bat
python fine_tuning.py <checkpoint> <dataset> <learning_rate> <batch_size>
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
python probing_comparison.py <finetuned model> <dataset> <classifier_type> <learning_rate> <batch_size>
```

Example:
```bat
python probing_comparison.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 1layer 1e-06 64
```

This generates results_probing.csv with the following data:

```
model_name, dataset_name, classifier_type, batch_size, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```

As explained in the paper, we only tested fine-tuned RoBERTa-large models in this and the next experiment, due to computational resources constrains. This model demonstrated a suitable balance between performance, consistency among datasets, and training time in the fine-tuning tests.


### Layerwise Probing
For the layerwise probing experiment (sec. 6), run:

Usage:
```bat
python probing_layer.py <finetuned model> <dataset> <classifier_type> <layer> <learning_rate> <batch_size>
```

Example:
```bat
python probing_layer.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 1layer 12 1e-06 64
```

This saves results in results_probing.csv containing the following data:

```
model_name, dataset_name, classifier_type, batch_size, layer, best_epoch, learning_rate, best_train_acc, best_train_loss.item(), best_validation_acc, best_validation_loss.item(), test_acc, test_loss.item(), f1_micro, f1_macro, f1_weighted
```

If you want to run multiple layers in a row, use the loop_layer.py script. By default, you only need to insert the finetuned model name, the probed dataset name, and the batch size. The script will loop for all layers, using two classifiers (1layer and 3layers) and two learning rates.

Usage:
```bat
python layer_loop.py <finetuned model> <dataset> <batch_size>
```

Example:
```bat
python layer_loop.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 64
```

## Citation
If you use or discuss this paper in your work, please cite it as follows:

```
@misc{pirozelli2023assessing,
      title={Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models}, 
      author={Paulo Pirozelli and Marcos M. José and Paulo de Tarso P. Filho and Anarosa A. F. Brandão and Fabio G. Cozman},
      year={2023},
      eprint={2312.11720},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements
This work was carried out at the Center for Artificial Intelligence at the University of São Paulo (C4AI-USP), with support by the São Paulo Research Foundation (FAPESP grant #2019/07665-4) and by the IBM Corporation.

![image](https://github.com/paulopirozelli/logicalreasoning/assets/39565459/3feca563-3699-40ba-aa23-814dcfb9929e)


