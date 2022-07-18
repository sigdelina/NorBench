# Updated code from NorBench

1. [x] Implementing XLM-R model for POS-tagging task for Bokmål and Nynorsk 

2. [x] Implementing XLM-R model for sentiment task

3. [ ] Updating script for automatical choosing model and classifier for the current task (POS, sentiment)


P.S. [Original code](https://github.com/ltgoslo/NorBERT/tree/main/benchmarking) is here.

|Task|Language|Model|Train Score|Dev Score|Test score|
|---|---|---|---|---|---|
|POS-tagging|Bokmål|xlm-roberta-base| 99.6 |97.7|97.5|
|POS-tagging|Nyrnorsk|xlm-roberta-base| 99.5 |97.5|97.3|
|Sentiment|Norwegian|xlm-roberta-base|77.1 (as best score) |72.5|71.8
