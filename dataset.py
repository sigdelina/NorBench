"""
Conllu --> tabulated file

Headers:
    - id:           id of each csv record
    - sentence:     sentence containing a word to be disambiguated (surrounded by 2 '[TGT]' tokens)
    - sense_keys:   list of candidate sense keys
    - targets:      list of indices of the correct sense keys (ground truths)
"""

import argparse
import conllu
import pandas as pd
from pathlib import Path
import conllu
import gzip
import re
import collections
import random
import pandas as pd
from wordnet import *
from tqdm.auto import tqdm
from tqdm.auto import tqdm
from collections import defaultdict


def read_conllu(path_to_conllu):

    with open(path_to_conllu, "r") as f:
        content = f.read()
                
    data = conllu.parse(content)

    return data

        


def main():
    
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_file",
        type=str,
        required=True,
        help="Path to conllu file with texts -- gold standart here with sense-annotated"
    )
    parser.add_argument(
        "--dir_to_rdf",
        type=str,
        required=True,
        help="Path to rdfs for Wordnet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory for suffient corpus in .csv."
    )
    parser.add_argument(
        "--max_num_gloss",
        type=int,
        default=4,
        help="Maximum number of candidate glosses from gold corpus"
    )
    parser.add_argument(
        "--use_augmentation",
        action='store_true',
        help="Whether to augment training dataset with example sentences from WordNet"
    )

    args = parser.parse_args()

    corpus_dir = Path(args.corpus_file)
    output_filename = "corpus"

    if args.max_num_gloss:
        output_filename += f"-max_num_gloss={args.max_num_gloss}"
    if args.use_augmentation:
        output_filename += "-augmented"
    csv_path = str(Path(args.output_dir).joinpath(f"{output_filename}.csv"))

    print("Data for gloss selection task was created...")


    sent_counter = 0
    WORDNET_POS = {'VERB': ['verb'], 'NOUN': ['noun', 'propn', 'subst'], 'ADJ': ['adj'], 'ADV': ['adv']}
    TGT_TOKEN = '[TGT]'

    id_dataset, sent_dataset, sense_key, targets = [], [], [], []

    wrds = args.dir_to_rdf + 'words.rdf'
    syns = args.dir_to_rdf + 'synsets.rdf'
    wordsn = args.dir_to_rdf +'wordsenses.rdf'
    hypn = args.dir_to_rdf + 'hyponymOf.rdf'
    synsets, wordsenses, words, hyponymy = parse_rdfs(syns, wordsn, wrds, hypn)
    graph_2 = wordsenses + words

    # get info from parsed file

    data = read_conllu(corpus_dir)

    print("Conllu file was parsed...")

    max_len_id = '0' * len(str(len(data)))
    for sentence in tqdm(data):

        token_count = 0
        tokens = [word['form'] for word in sentence]
        
        for word in sentence:

            # prepare ids for sentences
            id_param = 'd' + '0' * len(str(len(data))) + \
            '.s' + str('0' * len(str(len(data))) + str(sent_counter))[-len(str(len(data))):] + \
                '.t' + str('0' * len(str(len(data))) + str(token_count))[-len(str(len(data))):]
            
            try:
                synset_id = list(word['misc'].keys())
            except:
                synset_id = None

            tagged_sentence = " ".join(
                                tokens[:word['id']-1] + [TGT_TOKEN] + tokens[word['id']-1:word['id']] + [TGT_TOKEN] + tokens[word['id']:]
                            )
            
            cur_pos = [k for k, v in WORDNET_POS.items() if word['upos'] in v]

            if synset_id is not None and synset_id[0].isdigit():
                dict_synsets = graph_relations(wordsenses, synsets, graph_2, word, cur_pos[0], WORDNET_POS)
                counter_dict = [v for v in dict_synsets.values()][0]
                if len(counter_dict) == 0 and synset_id:
                    dict_synsets[word['lemma']] = synset_id

                senses = [v for v in dict_synsets.values()][0]
                remainder = args.max_num_gloss - len(senses)
                randome_senses = []

                if remainder < 0:
                    random_samples = [s for s in senses if s != synset_id[0]]
                    randomized = list(random.sample(random_samples, args.max_num_gloss - 1))
                    randomized.append(synset_id[0])
                    randome_senses = random.sample(randomized, len(randomized))
                else:
                    randome_senses = random.sample(senses, len(senses))
                
                gold_index = [randome_senses.index(synset_id[0])]
                
                id_dataset.append(id_param)
                sent_dataset.append(tagged_sentence)
                sense_key.append(randome_senses)
                targets.append(gold_index)
                
                token_count += 1
                
        sent_counter += 1
    
    dataset = pd.DataFrame({'id': id_dataset, 'sentence': sent_dataset, 'sense_key': sense_key, 'targets': targets})

    print("Dataset was created...")

    dataset.to_csv(csv_path)
    print("Dataset was saved to {csv_path}...")
    



if __name__ == '__main__':
    main()
