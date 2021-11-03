import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher

from aochildes.dataset import ChildesDataSet


transcripts = ChildesDataSet().load_transcripts()


nlp = spacy.load("en_core_web_sm", exclude=['ner', 'parser'])

matcher = Matcher(nlp.vocab)
pattern1 = [{'POS': 'NOUN'},
            {'LOWER': 'and'},
            {'LOWER': 'other'},
            {'POS': 'NOUN'},
            ]
pattern2 = [{'POS': 'NOUN'},
            {'IS_PUNCT': True, 'OP': '?'},
            {'LOWER': 'especially'},
            {'POS': 'NOUN'}]
pattern3 = [{'POS': 'NOUN'},
            {'IS_PUNCT': True, 'OP': '?'},
            {'LOWER': 'including'},
            {'POS': 'NOUN'}]
pattern4 = [{'POS': 'NOUN'},
            {'LOWER': 'or'},
            {'LOWER': 'other'},
            {'POS': 'NOUN'}]
pattern5 = [{'POS': 'NOUN'},
            {'IS_PUNCT': True, 'OP': '?'},
            {'LOWER': 'such'},
            {'LOWER': 'as'},
            {'POS': 'NOUN'}]

matcher.add("hypernyn-hyponym",
            [pattern1,
             pattern2,
             pattern3,
             pattern4,
             pattern5,
             ])

for doc in nlp.pipe(transcripts, n_process=4):
    doc: Doc
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        print(span.text)
