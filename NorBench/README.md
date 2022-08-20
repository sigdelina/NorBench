# Norbench

### Description


NorBench is a benchmark for testing language tasks for Norwegian languages.
At the moment, the benchmark is presented in 4 solved tasks. For each 5 models have already been implemented, and results for them were obtained.


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
   --- sentiments_finetuning.py
   --- ner_finetuning.py
       
```

### Part of Speech Tagging Task

For this task, [script](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/pos_finetuning.py) `pos_finetuning.py` should be used.

The input of the model is:

* `--model_name` - the name of the model with which the user intends to save the output file

* `--short_model_name` - the name of the model can be presented in several ways. 
  +  Firstly: in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + or the model name can be submitted as a full model name mentinoed in transformer library: `bert-base-multilingual-cased`

* `--training_language` - as POS-tagging task was analyzed for both Norwegian Bokmål and Norwegian Nynorsk, then `nob` or `nyr` should be used respectively

* `--epochs` - number of trainable epochs (`10` as default)


#### Running script

Trial `.slurm` file could be found in [experiments](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/slurm_example_nbbert_pos.slurm) folder

```
python3 pos_finetuning.py --model_name $NAME_OF_MODEL_1 --short_model_name $NAME_OF_MODEL_2 --training_language $LANGUAGE --epochs $EPOCHS

```


### Fine-grained Sentiment Analysis Task

The code and overall discription for the current task can be found by the [link](https://github.com/jerbarnes/sentiment_graphs)


### Binary Sentiment Analysis Task

For this task, [script](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/sentiment_finetuning.py) `sentiment_finetuning.py` should be used.

The input of the model is:

* `--model_name` - the name of the model with which the user intends to save the output file

* `--short_model_name` - the name of the model can be presented in several ways. 
  +  Firstly: in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + or the model name can be submitted as a full model name mentinoed in transformer library: `bert-base-multilingual-cased`
  
* `--use_class_weights` - a parameter that determines whether classes will be balanced when the model is running (classes are balanced when a `FALSE` value is passed to the parameter)

* `--training_language` - as POS-tagging task was analyzed for both Norwegian Bokmål and Norwegian Nynorsk, then `nob` or `nyr` should be used respectively

* `--epochs` - number of trainable epochs (`10` as default)


#### Running script

Trial `.slurm` file could be found in [experiments](https://github.com/sigdelina/NorBench/blob/main/NorBench/experiments/slurm_example_nbbert_sentiment.slurm) folder

```
python3 pos_sentiment.py --model_name $NAME_OF_MODEL_1 --short_model_name $NAME_OF_MODEL_2 --use_class_weights $WEIGHTED --training_language $LANGUAGE --epochs $EPOCHS
```

### Named Entitiy Recognition Task

!!! A fully assembled file in the specified format is being finalized (it will be ready by Monday, errors are possible now).

The previous version for this task is located at the [link](https://github.com/sigdelina/NorBench/tree/main/XLM-R%20model/ner).

---

The input of the model is: 

* `--model_type` - type of the model that one want to run (`bert`, `roberta`)
* `--model_name` - the name of the model can be presented in several ways.
  +  in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `xlm-roberta`
  + the model name can be submitted as a full model name mentinoed in transformer library: `xlm-roberta-base`
* `--dataset` - the part of name of the output file
* `--training_language` - as NER task was analyzed for both Norwegian Bokmål and Norwegian Nynorsk, then `nob` or `nyr` should be used respectively
* `--epochs` - number of trainable epochs (`20` as default)
