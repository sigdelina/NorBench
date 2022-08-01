# Updated code from NorBench

1. [x] Implementing XLM-R model for POS-tagging task for Bokmål and Nynorsk 

2. [x] Implementing XLM-R model for sentiment task

3. [ ] Updating script for automatical choosing model and classifier for the current task (POS, sentiment)

4. [ ] Trying several more models for current tasks

5. [ ] Implementing XLM-R model for NER task

    * Scores for NER task were obtained (the code differs from the origianl one and more suitable to one of the [oroginal scripts](https://github.com/sigdelina/NorBench/blob/main/original_code_norbench/experiments/ner.py)) -- organizing the script common with script of POS-tagging nad binary Sentimnet analysis is in progress.


P.S. [Original code](https://github.com/ltgoslo/NorBERT/tree/main/benchmarking) is here.

|Task|Language|Model|Train Score|Dev Score|Test score|
|---|---|---|---|---|---|
|POS-tagging|Bokmål|xlm-roberta-base| 99.6 |97.7|97.5|
|POS-tagging|Nyrnorsk|xlm-roberta-base| 99.5 |97.5|97.3|
|Sentiment|Norwegian|xlm-roberta-base|77.1 (as best score) |72.5|71.8|
