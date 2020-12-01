"""A simple example of extracting relations between phrases and entities using
spaCy's named entity recognizer and the dependency parse. Here, we extract
money and currency values (entities labelled as MONEY) and then check the
dependency tree to find the noun phrase they are referring to – for example:
$9.4 million --> Net income.

Compatible with: spaCy v2.0.0+
Last tested with: v2.2.1
"""
from __future__ import unicode_literals, print_function

import plac
import spacy
from spacy import displacy


TEXTS = [
    # "Joe Biden is happy, Donald Trump is sad"
    "Joe Biden to face test over access to sensitive information as he inherits Donald Trump's secret server"
    # "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]


@plac.annotations(
    model=("Model to load (needs parser and NER)", "positional", None, str)
)
def main(model="en_core_web_sm"):
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
    print("Processing %d texts" % len(TEXTS))

    for text in TEXTS:
        doc = nlp(text)
        print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
        relations = extract_currency_relations(doc)
        for r1, r2 in relations:
            print("{:<10}\t{}\t{}".format(r1.text, r2.ent_type_, r2.text))


def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def extract_currency_relations(doc):
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    # relations = []
    # for money in filter(lambda w: w.ent_type_ == "MONEY", doc):
    #     if money.dep_ in ("attr", "dobj"):
    #         subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
    #         if subject:
    #             subject = subject[0]
    #             relations.append((subject, money))
    #     elif money.dep_ == "pobj" and money.head.dep_ == "prep":
    #         relations.append((money.head.head, money))
    # return relations

    # displacy.serve(doc, style="dep")
    relations = []
    # for person in doc:and w.dep_ == "nsubj", doc
    for person in filter(lambda w: w.ent_type_ == "PERSON", doc):
    
    # for person in doc:
        relations.append((person.head, person))
        print("-----------------" ) 
        print("person: " + person.text) 
        print("person: " + person.dep_) 
        print("person: " + person.pos_) 
        # print("person.root: " + person.root.text) 
        # print("person.root: " + person.root.dep_) 
        # print("person.root.root: " + person.root.root.text) 
        # print("person.root.root: " + person.root.root.dep_) 
    
        # print("person.root.root.root: " + person.root.root.root.text) 
        # print("person.root.root.root: " + person.root.root.head.dep_) 
    for entity in doc.ents:
        print(entity.text, entity.label_)    
       # if person.dep_ in ("attr", "dobj"):
        #     subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
        #     if subject:
        #         subject = subject[0]
        #  relations.append((subject, money))
        # elif money.dep_ == "pobj" and money.head.dep_ == "prep":
    return relations


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Net income      MONEY   $9.4 million
    # the prior year  MONEY   $2.7 million
    # Revenue         MONEY   twelve billion dollars
    # a loss          MONEY   1b
