import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from collections import defaultdict
import numpy as np

from aochildes.dataset import ChildesDataSet
from aochildespatterns.utils import save_summary_to_txt

NUM_PARTS = 64
PATTERN_NAME = 'adnominal'

transcripts_ = ChildesDataSet().load_transcripts()

# make num transcripts divisible by NUM_PARTS
transcripts = transcripts_[:len(transcripts_) - len(transcripts_) % NUM_PARTS]

nlp = spacy.load("en_core_web_sm", exclude=['ner'])

matcher = Matcher(nlp.vocab)
pattern = [{'DEP': 'amod', 'OP': "+"},  # adjectival modifier
           {'POS': 'NOUN'},
           ]
matcher.add(PATTERN_NAME,
            [pattern],
            )

transcript2spans = defaultdict(list)
y_ = np.zeros(len(transcripts))
for n, doc in enumerate(nlp.pipe(transcripts, n_process=4)):
    doc: Doc
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        transcript2spans[n].append(span.text)

    num_spans_found = len(transcript2spans[n])
    y_[n] = num_spans_found
    print(f'Transcript {n:>6,} | Found {num_spans_found :>6,} {PATTERN_NAME} patterns')

y = y_.reshape((NUM_PARTS, -1)).sum(axis=1)
x = [i for i in range(len(y))]
print(x)
print(y)
print(y.shape)

save_summary_to_txt(x=x,
                    y=y,
                    quantity_name=f'num_{PATTERN_NAME}',
                    )
