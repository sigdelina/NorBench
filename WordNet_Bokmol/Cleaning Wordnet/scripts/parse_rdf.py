# coding=utf8

# Under revision

import sys
import argparse
import rdflib
from rdflib import URIRef
from rdflib.namespace import RDF
   

# Some useful URIs.

hyp = URIRef("http://www.wordnet.dk/owl/instance/2009/03/schema/hyponymOf")
word = "http://www.w3.org/2006/03/wn/wn20/schema/word"
wordsense = "http://www.w3.org/2006/03/wn/wn20/schema/containsWordSense"
lexform = "http://www.w3.org/2006/03/wn/wn20/schema/lexicalForm"
nounURI = "http://www.w3.org/2006/03/wn/wn20/schema/NounWordSense"
verbURI = "http://www.w3.org/2006/03/wn/wn20/schema/VerbWordSense"
adjURI = "http://www.w3.org/2006/03/wn/wn20/schema/AdjectiveWordSense"
lexform = "http://www.w3.org/2006/03/wn/wn20/schema/lexicalForm"
word = "http://www.w3.org/2006/03/wn/wn20/schema/word"


def parsing(synsets, wordsenses, words, hyponymy):
	lookup_id = {}
	lookup_string ={}
	unique_lex = set()

	# Iterating over the words and their wordsenses, 
	# and map word IDs to the lexical forms and vice versa.

	for s, p, o in words.triples((None, URIRef(lexform), None)):
		for s2, p2, o2 in wordsenses.triples((None, URIRef(word), s)):
			if " " in o:
				a = "_".join(str(o.encode("utf-8")).split())
			else:
				a = str(o.encode("utf-8"))

			if (s2, None, URIRef(nounURI)) in wordsenses:
				# print(a)
				a += "_" + "subst"
			elif (s2, None, URIRef(verbURI)) in wordsenses:
				a += "_" + "verb"
			elif (s2, None, URIRef(adjURI)) in wordsenses:
				a += "_" + "adj"
				
				if a not in unique_lex:
					unique_lex.add(a)
					lookup_id[a] = set()
				lookup_id[a].add(s)
				lookup_string[s] = a

	return lookup_id, lookup_string, unique_lex


def main():
	
	parser = argparse.ArgumentParser()
    
	parser.add_argument(
        "--synsets",
        type=str,
        required=True,
        help="Path to .rdf file with synsets"
    )
	parser.add_argument(
        "--wordsenses",
        type=str,
        required=True,
        help="Path to .rdf file with wordsenses"
    )
	parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Path to .rdf file with words"
    )
	parser.add_argument(
        "--hyponym",
        type=str,
        required=True,
        help="Path to .rdf file with hyponym"
    )

	args = parser.parse_args()

	synsets = rdflib.Graph()
	wordsenses = rdflib.Graph()
	words = rdflib.Graph()
	hyponymy = rdflib.Graph()
	wordsenses = rdflib.Graph()	

	synsets.parse(args.synsets)
	wordsenses.parse(args.wordsenses)
	words.parse(args.words)
	hyponymy.parse(args.hyponym)
	synsets.parse(args.hyponym) # merging synsets and hyponymOf  

	lookup_id, lookup_string, unique_lex = parsing(synsets, wordsenses, words, hyponymy)
	
	return lookup_id, lookup_string, unique_lex 


if __name__ == '__main__':
    main()