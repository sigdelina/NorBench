#!/bin/env python3

from transformers import XLMRobertaForTokenClassification, BertForTokenClassification,  XLMRobertaTokenizerFast, BertTokenizer


class BERTTokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels):
        # This propogates the label over any subwords that
        # are created by the byte-pair tokenization for training

        # IMPORTANT: For testing, you will have to undo this step by combining
        # the subword elements, and

        split_tokens, split_labels = [], []
        idx_map = []
        for ix, token in enumerate(tokens):
            sub_tokens = self.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map


class XLMRTokenizer(XLMRobertaTokenizerFast):
    def subword_tokenize(self, tokens, labels):
        # This propogates the label over any subwords that
        # are created by the byte-pair tokenization for training

        # IMPORTANT: For testing, you will have to undo this step by combining
        # the subword elements, and

        split_tokens, split_labels = [], []
        idx_map = []
        for ix, token in enumerate(tokens):
            sub_tokens = self.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map


models_type = {
    "bert": {
        "model": BertForTokenClassification.from_pretrained,
        "tokenizer": BERTTokenizer.from_pretrained,
        "model_names": {
            "bert-base-multilingual-cased": "bert-base-multilingual-cased",
            "mbert": "bert-base-multilingual-cased",
            "norbert": "ltgoslo/norbert",
            "norbert2": "ltgoslo/norbert2",
            "nb-bert-base": "NbAiLab/nb-bert-base"
        }
    },
    "roberta": {
        "model": XLMRobertaForTokenClassification.from_pretrained,
        "tokenizer": XLMRTokenizer.from_pretrained,
        "model_names": {
            "tf-xlm-roberta-base": "jplu/tf-xlm-roberta-base",
            "xlm-roberta-base": "xlm-roberta-base",
        }
    }
}


def get_ner_tags():
    return ['O', 'B-PER', 'B-LOC',  'I-PER',  'B-PROD',  'B-GPE_LOC',  'I-PROD', 
            'B-DRV',  'I-DRV',  'B-EVT',  'I-EVT',  'B-ORG',  'I-LOC',  'I-GPE_LOC',
            'I-ORG',  'B-GPE_ORG',  'I-GPE_ORG',  'B-MISC', 'I-MISC']
        

def organized_subsets(data, id2label):
    ids, tokens, tags = [], [], []
    for el in data:
        ids.append(el['id'])
        tokens.append(el['tokens'])
        tags.append([id2label[tag] for tag in el['ner_tags']])
    return ids, tokens, tags


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs