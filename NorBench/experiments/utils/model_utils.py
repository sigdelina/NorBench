#!/bin/env python3

import tensorflow as tf
# import sentencepiece
from transformers import (TFBertForSequenceClassification, BertTokenizer, AutoTokenizer,
                          TFXLMRobertaForSequenceClassification, XLMRobertaTokenizer, TFAutoModelForSequenceClassification,
                          TFBertForTokenClassification, TFXLMRobertaForTokenClassification, TFAutoModelForTokenClassification,
                          TFDistilBertForTokenClassification, TFDistilBertForSequenceClassification, DistilBertTokenizer)

from data_preparation.data_preparation_pos import *


models = {
    "bert": {
        "pos": TFBertForTokenClassification.from_pretrained,
        "sentiment": TFBertForSequenceClassification.from_pretrained,
    },
    "xlm-roberta": {
        "pos": TFXLMRobertaForTokenClassification.from_pretrained,
        "sentiment": TFXLMRobertaForSequenceClassification.from_pretrained,
    },
    "distilbert": {
        "pos": TFDistilBertForTokenClassification.from_pretrained,
        "sentiment": TFDistilBertForSequenceClassification.from_pretrained,
    },
    "auto": {
        "pos": TFAutoModelForTokenClassification.from_pretrained,
        "sentiment": TFAutoModelForSequenceClassification.from_pretrained
    }
}

tokenizers = {
    "bert": {
        "pos": MBERTTokenizer.from_pretrained,
        "sentiment": BertTokenizer.from_pretrained
    },
    "xlm-roberta": {
        "pos": XLMRTokenizer.from_pretrained,
        "sentiment": XLMRobertaTokenizer.from_pretrained
    },
    "distilbert": {
        "pos": DisBerTokenizer.from_pretrained,
        "sentiment": DistilBertTokenizer.from_pretrained
    },
    "auto": {
        "pos": AutoMTokenizer.from_pretrained,
        "sentiment": AutoTokenizer.from_pretrained
    }
}

# new models can be added to the current dictionary
# or even if model is not added to the dictionary it can be still used  
model_names = {
        "bert-base-multilingual-cased": "bert-base-multilingual-cased",
        "tf-xlm-roberta-base": "jplu/tf-xlm-roberta-base",
        "mbert": "bert-base-multilingual-cased",
        "norbert1": "ltgoslo/norbert",
        "norbert": "ltgoslo/norbert",
        "norbert2": "ltgoslo/norbert2",
        "xlm-roberta-base": "xlm-roberta-base",
        "xlm-roberta": "xlm-roberta-base",
        "nb-bert-base": "NbAiLab/nb-bert-base",
        "distilbert": "distilbert-base-uncased"
    }

def set_tf_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def get_full_model_names(short_model_name, d=model_names):
    if short_model_name in d.values():
        return d[short_model_name] if short_model_name in d else short_model_name
    else:
        if short_model_name in d:
          model_name = d[short_model_name]
          return d[model_name] if model_name in d else model_name
        else:
          return short_model_name


def get_name_from_dict_keys(short_model_name, d=model_names):
    
    if short_model_name not in d and short_model_name not in d.values():
      short_name = short_model_name

    if short_model_name in d:
        short_name = short_model_name
    elif short_model_name in d.values():
        reverse_lookup = dict([(val, key) for key, val in d.items()])
        short_name = reverse_lookup[short_model_name]

    if 'roberta' in short_name:
      return "xlm-roberta"
    elif 'distilbert' in short_name:
      return 'distilbert'
    elif 'bert' in short_name:
      return 'bert'
    else:
      return 'auto'


def create_model(short_model_name, task, num_labels, from_pt=False):
    short_name = get_name_from_dict_keys(short_model_name)
    model = models[short_name][task](get_full_model_names(short_model_name), num_labels=num_labels)
    tokenizer = get_tokenizer(short_model_name, short_name, task)
    return model, tokenizer


def get_tokenizer(short_model_name, short_name, task, do_lower_case=False):
    if tokenizers[short_name][task] is not None:
        try:
            return tokenizers[short_name][task](get_full_model_names(short_model_name))
        except:
            return tokenizers[short_name][task](get_full_model_names(short_model_name), do_lower_case)
    else:
        try:
            return tokenizers['auto'][task](get_full_model_names(short_model_name))
        except:
            return tokenizers['auto'][task](get_full_model_names(short_model_name), do_lower_case)


def compile_model(model, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)        
    return model


def make_batches(dataset, batch_size, repetitions, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(int(1e6), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    n_batches = len(list(dataset.as_numpy_iterator()))
    dataset = dataset.repeat(repetitions)
    return dataset, n_batches
