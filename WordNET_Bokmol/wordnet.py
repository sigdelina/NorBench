import re
import rdflib
from rdflib import Graph
from rdflib import URIRef
from rdflib.namespace import RDF


def parse_rdfs(synsets_file, wordsenses_file, words_file, hyponymy_file):

    synsets = rdflib.Graph()
    print('Synsets graph was initialized ... ')

    wordsenses = rdflib.Graph()
    print('Wordsenses graph was initialized ... ')

    words = rdflib.Graph()
    print('Words graph was initialized ... ')

    hyponymy = rdflib.Graph()
    print('Hyponymys graph was initialized ... ')

    wordsenses = rdflib.Graph()
    print('Wordsenses graph was initialized ... ')
    

    synsets.parse(synsets_file)
    print('Synsets graph was created ... ')

    wordsenses.parse(wordsenses_file)
    print('Wordsenses graph was created ... ')

    words.parse(words_file)
    print('Words graph was created ... ')

    hyponymy.parse(hyponymy_file)
    print('Hyponymys graph was created ... ')

    synsets.parse(hyponymy_file)
    print('Synsets graph was created ... ')

    return synsets, wordsenses, words, hyponymy


def graph_relations(wordsenses, synsets, words_syns, word, pos, WORDNET_POS):

    word_dict = {}
    noun_list, verb_list, adj_list, adv_list = [], [], [], []
    lexem = word['lemma']
    main_syn = list(word['misc'].keys())[0]


    nounURI = "http://www.w3.org/2006/03/wn/wn20/schema/NounSynset"
    verbURI = "http://www.w3.org/2006/03/wn/wn20/schema/VerbSynset"
    adjURI = "http://www.w3.org/2006/03/wn/wn20/schema/AdjectiveSynset"
    advURI = "http://www.w3.org/2006/03/wn/wn20/schema/AdverbSynset"

    # all synsets for current word
    qres = words_syns.query("""
        PREFIX  wn20schema: <http://www.w3.org/2006/03/wn/wn20/schema/>

        SELECT  ?aSynset
        WHERE   { ?aSynset wn20schema:containsWordSense ?aWordSense .
                ?aWordSense wn20schema:word ?aWord .
                ?aWord wn20schema:lexicalForm  "%s"@nb		
        }"""%(lexem))
    
    for row in qres:

        if (URIRef(row[0]), None, URIRef(nounURI)) in synsets:
            noun_list.append(re.search(r'synset-(\d+)$', str(row[0])).group(1))
        elif (URIRef(row[0]), None, URIRef(verbURI)) in synsets:
            verb_list.append(re.search(r'synset-(\d+)$', str(row[0])).group(1))
        elif (URIRef(row[0]), None, URIRef(adjURI)) in synsets:
            adj_list.append(re.search(r'synset-(\d+)$', str(row[0])).group(1))
        elif (URIRef(row[0]), None, URIRef(advURI)) in synsets:
            adv_list.append(re.search(r'synset-(\d+)$', str(row[0])).group(1))


    cur_list = []
    if pos == 'VERB':
        cur_list = verb_list
    elif pos == 'NOUN':
        cur_list = noun_list
    elif pos == 'ADJ':
        cur_list = adj_list
    elif pos == 'ADV':
        cur_list = adv_list
    
    if main_syn not in cur_list:
        cur_list.append(main_syn)
        
    word_dict[lexem] = list(set(cur_list))
    return word_dict
