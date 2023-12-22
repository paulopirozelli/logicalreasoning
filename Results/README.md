The Excel file contains the full set of results for the experiments described in the paper. Best results are highlighted in bold.

The file is organized according to the following tabs:

**FT - Logic:** Results for the hypothesis classification task [Sec. 4, Table 2]. Each of the 21 language models is fine-tuned on the 4 logical reasoning datasets (FOLIO, LogicNLI, RuleTaker, SimpleLogic), using 2 different learning rates (1e-06, 1e-05). The classifier is a 1layer head. Number of tests: 168.

**Probing - FO, Probing - LN, Probing - RT, Probing - SL**: This set of tabs provides the results of the cross-probing task on the 4 different logical reasoning datasets (FOLIO, LogicNLI, RuleTaker, SimpleLogic) [Sec. 5, Table 3]. Each tab reports the scores for 5 language models (the best RoBERTa-large model for each dataset + a pretrained RoBERTa) on the specific dataset, using 2 probes (1layer, 3layers) and 2 learning rates (1e-06, 1e-05). Number of tests: 20 per dataset.

**Layer - FO, Layer - LN, Layer - RT, Layer - SL:** The tabs provide the results of the layerwise probing task on the 4 different logical reasoning datasets (FOLIO, LogicNLI, RuleTaker, SimpleLogic) [Sec. 6, Figure 2]. Each tab reports the scores for the best fine-tuned RoBERTa model on the specific dataset probed layerwise. We use 2 probes (1layer, 3layers) and 2 learning rates (1e-06, 1e-05). Number of tests: 100 per dataset.
