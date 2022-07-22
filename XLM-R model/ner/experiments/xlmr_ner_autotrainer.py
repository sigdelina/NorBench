from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, DataCollatorForTokenClassification, Trainer, TrainingArguments
import glob
import argparse
import data_preparation.data_preparation_pos as data_prepare
from transformers.data.processors.utils import InputFeatures
from utils.utils import read_conll
import numpy as np
import seqeval
import datasets
from datasets import load_metric, load_dataset, Dataset
metric = load_metric("seqeval")
import warnings
warnings.filterwarnings("ignore")


training_language = 'nno'
run_name = 'xlm-roberta'
model_identifier = 'xlm-roberta-base'
current_task = "ner"
epochs = 10
max_length = 256

def get_ner_tags():
    return ['O', 'B-PER', 'B-LOC',  'I-PER',  'B-PROD',  'B-GPE_LOC',  'I-PROD', 
            'B-DRV',  'I-DRV',  'B-EVT',  'I-EVT',  'B-ORG',  'I-LOC',  'I-GPE_LOC',
            'I-ORG',  'B-GPE_ORG',  'I-GPE_ORG',  'B-MISC', 'I-MISC']

tagset = get_ner_tags()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_steps=3000,
    save_strategy='steps',
    load_best_model_at_end = True, 
    save_total_limit=3,
)

def load_dataset_ner(lang_path, dataset_name="test"):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""

    data = read_conll(glob.glob(lang_path + "{}.conllu".format(dataset_name.split("_")[0]))[0], label_nr=9)

    examples = [{"id": sent_id, "tokens": tokens, "ner_tags": [tag.split("|")[-1].split('=')[1] for tag in tags]} for sent_id, tokens, tags in
                zip(data[0], data[1], data[2])]
    
    return examples


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


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [tagset[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tagset[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def organized_subsets(data, id2label):
    ids, tokens, tags = [], [], []
    for el in data:
        ids.append(el['id'])
        tokens.append(el['tokens'])
        tags.append([id2label[tag] for tag in el['ner_tags']])
    return ids, tokens, tags


def collecting_data(tokenizer, path):
    
    train_data = load_dataset_ner(
                        path+'no_nynorsk-ud-', dataset_name='train'
                    )
    dev_data = load_dataset_ner(
                        path+'no_nynorsk-ud-', dataset_name='dev'
                    )
    test_data = load_dataset_ner(
                        path+'no_nynorsk-ud-', dataset_name='test'
                    )

    print(train_data[15])

    # dict_sets = {'train': train_data, 'dev': dev_data, 'test': test_data}
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {t: i for i, t in enumerate(tagset)}

    tr_ids, tr_tokens, tr_tags = organized_subsets(train_data, id2label)

    de_ids, de_tokens, de_tags = organized_subsets(dev_data, id2label)

    te_ids, te_tokens, te_tags = organized_subsets(test_data, id2label)


    data = datasets.DatasetDict({'train': Dataset.from_dict({'id': tr_ids,'tokens': tr_tokens, 'tags': tr_tags}),
                                'dev': Dataset.from_dict({'id': de_ids,'tokens': de_tokens, 'tags': de_tags}),
                                'test': Dataset.from_dict({'id': te_ids,'tokens': te_tokens, 'tags': te_tags})})
    
    print(data['train'][15])


    tokenized_data = data.map(tokenize_and_align_labels, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    
    return tokenized_data, data_collator


def compute_predictions(trainer, tagset, tokenized_data):

    predictions, labels, _ = trainer.predict(tokenized_data["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [tagset[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tagset[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

def main():

    # parser = argparse.ArgumentParser()

    pos_data_path = "../data/ner/nno/"

    model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(tagset))
    tokenizer = data_prepare.XLMRTokenizer.from_pretrained('xlm-roberta-base')
    tokenized_data, data_collator = collecting_data(tokenizer, pos_data_path)
    
    print('---Data was collected---')
    print(tokenized_data)
    print(data_collator)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print('TRAINING')
    trainer.train()
    
    print('DEV')
    trainer.evaluate()
    print(f"Scores on dev dataset: {trainer}")
    
    print('TESTING')

    predictions = compute_predictions(trainer, tokenized_data)
    print(f"Scores on test dataset: {predictions}")


if __name__ == "__main__":
    main()


