import spacy
import json
from spacy import displacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")

text = ("Thessaloniki is the best place you can be in 2020, the year of crisis because of corona.")
text = ("Joe Biden to face test over access to sensitive information as he inherits Donald Trump's secret server")

doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

print(type(doc.ents))
print(doc.ents)

# json_doc = doc.to_json()
# print(type(json_doc))
# print(json_doc)


ent_list = []
# Find named entities, phrases and concepts
for entity in doc.ents:
    # ent_list.append((str(entity.text), str(entity.label_)))
    # print(type(entity.label_))
    print(entity.text, entity.label_)
displacy.serve(doc, style="dep")
# print(type(ent_list[0]))
# print(ent_list)
# json_string = json.dumps(ent_list)
# print(type(json_string))
# print(json_string)


