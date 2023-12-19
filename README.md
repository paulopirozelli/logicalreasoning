This is the repository for the paper "Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models".

**Abstract:** Logical reasoning is central to complex human activities, such as thinking, debating, and planning; it is also a central component of many AI systems as well. In this paper, we investigate the extent to which encoder-only transformer language models (LMs) can reason according to logical rules. We ask whether those LMs can deduce theorems in propositional calculus and first-order logic; if their relative success in these problems reflects general logical capabilities; and which layers contribute the most to the task. First, we show for several encoder-only LMs that they can be trained, to a reasonable degree, to determine logical validity on various datasets. Next, by cross-probing fine-tuned models on these datasets, we show that LMs have difficulty in transferring their putative logical reasoning ability, which suggests that they may have learned dataset-specific features, instead of a general capability. Finally, we conduct a layerwise probing experiment, which shows that the hypothesis classification task is mostly solved through higher layers.

## Results 
Full results for the experiments are found in the sheet [results.xlsx](https://github.com/paulopirozelli/logicalreasoning/blob/main/results.xlsx). 


## Datasets
Datasets are store in a Google Drive [folder](https://drive.google.com/drive/folders/1YpRoveEJJZIOUyAMeeo5LF6kt8eAFkya). 

Appendix A of the paper describes the datasets in detail and gives the original sources.

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

As explained in the paper, we only tested fine-tuned RoBERTa-large models in this and the next experiments, due to computational resources constrains. This model demonstrated a suitable balance between performance, consistency among datasets, and training time in the fine-tuning tests.


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

If you want to run multiple layers in a row, use the loop_layer.py script. By default, you only need to insert the finetuned model name, the probed dataset name, and the batch size. The script will loop for all layers, using two classifiers (1layer and 3layers) and two learning rates.

Usage:
```bat
python layer_loop.py <fine-tuned_model> <dataset> <batch_size>
```

Example:
```bat
python layer_loop.py bert-base-uncased_FOLIO_2_1e-06.pth FOLIO 64
```

## Citation
If you use or discuss this paper in your work, please cite it as follows:

```
@inproceedings{10.1145/3459637.3482012,
author = {Paschoal, Andr\'{e} F. A. and Pirozelli, Paulo and Freire, Valdinei and Delgado, Karina V. and Peres, Sarajane M. and Jos\'{e}, Marcos M. and Nakasato, Fl\'{a}vio and Oliveira, Andr\'{e} S. and Brand\~{a}o, Anarosa A. F. and Costa, Anna H. R. and Cozman, Fabio G.},
title = {Pir\'{a}: A Bilingual Portuguese-English Dataset for Question-Answering about the Ocean},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482012},
doi = {10.1145/3459637.3482012},
abstract = {Current research in natural language processing is highly dependent on carefully produced
corpora. Most existing resources focus on English; some resources focus on languages
such as Chinese and French; few resources deal with more than one language. This paper
presents the Pir\'{a} dataset, a large set of questions and answers about the ocean and
the Brazilian coast both in Portuguese and English. Pir\'{a} is, to the best of our knowledge,
the first QA dataset with supporting texts in Portuguese, and, perhaps more importantly,
the first bilingual QA dataset that includes this language. The Pir\'{a} dataset consists
of 2261 properly curated question/answer (QA) sets in both languages. The QA sets
were manually created based on two corpora: abstracts related to the Brazilian coast
and excerpts of United Nation reports about the ocean. The QA sets were validated
in a peer-review process with the dataset contributors. We discuss some of the advantages
as well as limitations of Pir\'{a}, as this new resource can support a set of tasks in
NLP such as question-answering, information retrieval, and machine translation.},
booktitle = {Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
pages = {4544–4553},
numpages = {10},
keywords = {Portuguese-English dataset, question-answering dataset, bilingual dataset, ocean dataset},
location = {Virtual Event, Queensland, Australia},
series = {CIKM '21}
}
```

## Acknowledgements
This work was carried out at the Center for Artificial Intelligence at the University of São Paulo (C4AI-USP), with support by the São Paulo Research Foundation (FAPESP grant #2019/07665-4) and by the IBM Corporation.

![image](https://github.com/paulopirozelli/logicalreasoning/assets/39565459/3feca563-3699-40ba-aa23-814dcfb9929e)


