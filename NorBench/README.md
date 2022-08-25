# Norbench

### Description


NorBench is a benchmark for testing language tasks for Norwegian languages.
At the moment, the benchmark is presented in [4 solved tasks](http://wiki.nlpl.eu/Vectors/norlm/norbert). For each at least 5 models have already been implemented, and results for them were obtained.

In the current documentation information about 3 of 4 tasks is provided in details:

+ [Part-Of-Speech tagging task](#POS)
  - [Parameters](#POS_PARAMS)
  - [How to run the training script](#POS_SCRIPT)
  - [Evaluation](#POS_EVAL)
  - [Available Models](#POS_MODELS)
+ [Fine-grained Sentiment Analysis task](#FINEGRAINED) -- detailed information about the current task is provided in the repository by link in the section
+ [Binary Snetiment Analysis task](#BINARYSENT)
  - [Parameters](#BINARYSENT_PARAMS)
  - [How to run the training script](#BINARYSENT_SCRIPT)
  - [Evaluation](#BINARYSENT_EVAL)
  - [Available Models](#BINARYSENT_MODELS)
+ [Named Entity Recognition task](#NER)
  - [Parameters](#NER_PARAMS)
  - [How to run the training script](#NER_SCRIPT)
  - [Evaluation](#NER_EVAL)
  - [Available Models](#NER_MODELS)
  

### Structure

The current repository contains the `experiments` folder with run scripts and `data` folder with training, vaidation, and test sets 

The structure of `data` is following:

```
--- data
  |
   --- pos
      |
       --- nob
           ...
       --- nyr
           ...
  
   --- sentiment
       |
        --- no
            ...
   --- ner
       |
        --- nob
            ...
        --- nyr
            ...
    
```

The structure of `experiments` is following:

```
--- experiments
  |
   --- data_preparation
       ... (files with data prepared for the current task)
  
   --- utils
       ... (general files)
       
   --- results
       ...
   
   --- pos_finetuning.py
   --- sentiment_finetuning.py
   --- ner_finetuning.py
       
```

---

### <a name="POS"></a> Part of Speech Tagging Task


For this task, [script](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/pos_finetuning.py) `pos_finetuning.py` should be used.



#### <a name="POS_PARAMS"></a>  Parameters

The input of the model is:

* `--model_name` - the name of the model with which the user intends to save the output file

* `--short_model_name` - the name of the model can be presented in several ways. 
  +  Firstly: in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + or the model name can be submitted as a full model name mentinoed in transformer library: `bert-base-multilingual-cased`

* `--training_language` - as POS-tagging task was analyzed for both Norwegian Bokmål and Norwegian Nynorsk, then `nob` or `nyr` should be used respectively

* `--epochs` - number of trainable epochs (`10` as default)


#### <a name="POS_SCRIPT"></a> Running script


Scripts could be run on the [SAGA SIGMA](https://documentation.sigma2.no/index.html)

In order to run the script on Saga, it is necessary to put arguments for [parameters](#POS_PARAMS) in the form indicated below.


Trial `.slurm` file could be found in [experiments](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/slurm_example_nbbert_pos.slurm) folder

```
python3 pos_finetuning.py --model_name $NAME_OF_MODEL_1 --short_model_name $NAME_OF_MODEL_2 --training_language $LANGUAGE --epochs $EPOCHS

```

#### <a name="POS_EVAL"></a>  Evaluation

Accuracy is used to perform the evaluation of the current task. The calculation of the metric takes place inside the script, so the user receives the table with the accuracy scores obtained on the validation subset and on the testing data. The table with output scores is automatically stored in the RESULTS folder.

The [final table](http://wiki.nlpl.eu/Vectors/norlm/norbert) includes the results of accuracy score on the testing dataset.


#### <a name="POS_MODELS"></a>  Models that have been successfully tested on this script

Currently, this script can work with Bert-Like-Models, DistilBert, and Xlm-Roberta models.

The use of other models in this benchmark is in the process of being resolved.

The list below describes the models for which it was possible to successfully obtain scores until now:

- mBERT: `bert-base-multilingual-cased`
- XLM-R: `xlm-roberta-base`
- NorBERT: `ltgoslo/norbert`
- NorBERT2: `ltgoslo/norbert2`
- NB-BERT-Base: `NbAiLab/nb-bert-base`
- Notram: `NbAiLab/notram-bert-norwegian-uncased-080321`
- Distilbert: `distilbert-base-uncased` -- there is little sense in using this model, however, an attempt was made to launch 
- ScandiBERT: `vesteinn/ScandiBERT` -- IN PROGRESS

---

### <a name="FINEGRAINED"></a> Fine-grained Sentiment Analysis Task

The code and overall discription for the current task can be found by the [link](https://github.com/jerbarnes/sentiment_graphs)

---

### <a name="BINARYSENT"></a> Binary Sentiment Analysis Task

For this task, [script](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/sentiment_finetuning.py) `sentiment_finetuning.py` should be used.

#### <a name="BINARYSENT_PARAMS"></a>  Parameters

The input of the model is:

* `--model_name` - the name of the model with which the user intends to save the output file

* `--short_model_name` - the name of the model can be presented in several ways. 
  +  Firstly: in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + or the model name can be submitted as a full model name mentinoed in transformer library: `bert-base-multilingual-cased`
  
* `--use_class_weights` - a parameter that determines whether classes will be balanced when the model is running (classes are balanced when a `FALSE` value is passed to the parameter)

* `--training_language` - as POS-tagging task was analyzed for both Norwegian Bokmål and Norwegian Nynorsk, then `nob` or `nyr` should be used respectively

* `--epochs` - number of trainable epochs (`10` as default)


#### <a name="BINARYSENT_SCRIPT"></a> Running script

Scripts could be run on the [SAGA SIGMA](https://documentation.sigma2.no/index.html)

In order to run the script on Saga, it is necessary to put arguments for [parameters](#BINARYSENT_PARAMS) in the form indicated below.


Trial `.slurm` file could be found in [experiments](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/slurm_example_nbbert_sentiment.slurm) folder

```
python3 sentiment_finetuning.py --model_name $NAME_OF_MODEL_1 --short_model_name $NAME_OF_MODEL_2 --use_class_weights $WEIGHTED --training_language $LANGUAGE --epochs $EPOCHS
```

#### <a name="BINARYSENT_EVAL"></a>  Evaluation

F1 score is used to perform the evaluation of the current task. The calculation of the metric takes place inside the script, so the user receives the table with the F1 scores obtained on the validation subset and on the testing data. The table with output scores is automatically stored in the RESULTS folder.

The [final table](http://wiki.nlpl.eu/Vectors/norlm/norbert) includes the results of F1 score on the testing dataset.


#### <a name="BINARYSENT_MODELS"></a>  Models that have been successfully tested on this script

Currently, this script can work with Bert-Like-Models, DistilBert, Xlm-Roberta models and models which are supported by AutoModel.from_pretrained by transormers library.

The use of other models in this benchmark is in the process of being resolved.

The list below describes the models for which it was possible to successfully obtain scores until now:

- mBERT: `bert-base-multilingual-cased`
- XLM-R: `xlm-roberta-base`
- NorBERT: `ltgoslo/norbert`
- NorBERT2: `ltgoslo/norbert2`
- NB-BERT-Base: `NbAiLab/nb-bert-base`
- Notram: `NbAiLab/notram-bert-norwegian-uncased-080321`
- XLM: `xlm-mlm-100-1280` -- was selected to test the possibility of launching via AutoModels
- Distilbert: `distilbert-base-uncased` -- there is little sense in using this model, however, an attempt was made to launch 
- ScandiBERT: `vesteinn/ScandiBERT` -- IN PROGRESS

---

### <a name="NER"></a> Named Entitiy Recognition Task

!!! A fully assembled file in the specified format is being finalized (it will be ready by Monday, errors are possible now).

The previous version for this task for `xlm-roberta-base` is located at the [link](https://github.com/sigdelina/NorBench/tree/main/XLM-R%20model/ner), while the version for `bert` is located by the [link](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/bert_ner.py).

---

For this task, [script](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/ner_finetuning.py) `ner_finetuning.py` should be used.

#### <a name="NER_PARAMS"></a>  Parameters

The input of the model is: 

* `--model_type` - type of the model that one want to run (`bert`, `roberta`)
* `--model_name` - the name of the model can be presented in several ways.
  +  in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `xlm-roberta`
  + the model name can be submitted as a full model name mentinoed in transformer library: `xlm-roberta-base`
* `--dataset` - the part of name of the output file
* `--training_language` - as NER task was analyzed for both Norwegian Bokmål and Norwegian Nynorsk, then `nob` or `nyr` should be used respectively
* `--epochs` - number of trainable epochs (`20` as default)


#### <a name="NER_SCRIPT"></a> Running script


Scripts could be run on the [SAGA SIGMA](https://documentation.sigma2.no/index.html)

In order to run the script on Saga, it is necessary to put arguments for [parameters](#NER_PARAMS) in the form indicated below.


Trial `.slurm` file could be found in [experiments](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/slurm_example_nob_norbert_ner.slurm) folder

```
python3 ner_finetuning.p --model_type $MODEL_TYPE --model_name $NAME_OF_MODEL_1 --dataset $NAME_OF_DATASET --training_language $LANGUAGE --epochs $EPOCHS
```

#### <a name="NER_EVAL"></a>  Evaluation




#### <a name="NER_MODELS"></a>  Models that have been successfully tested on this script

