import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from typing import List, Generator

from aochildes.dataset import ChildesDataSet
from aochildespatterns.utils import save_summary_to_txt
from aochildespatterns.probes import probes

NUM_PARTS = 64
PATTERN_NAME = 'amod+PROBE'

transcripts_ = ChildesDataSet().load_transcripts()

# make num transcripts divisible by NUM_PARTS
transcripts = transcripts_

nlp = spacy.load("en_core_web_sm", exclude=['ner'])

matcher = Matcher(nlp.vocab)

pattern = [{'DEP': 'amod', 'OP': "+"},  # adjectival modifier,
           {"LEMMA": {"IN": probes}},
           ]

matcher.add(PATTERN_NAME,
            [pattern],
            )


def gen_spans_by_partition(texts: List[str]) -> Generator[List[str], None, None]:
    num_ts_in_part = len(texts) // NUM_PARTS

    spans = []
    for transcript_id, doc in enumerate(nlp.pipe(texts, n_process=4)):

        doc: Doc
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            print(span.text)
            spans.append(span.text)

        # yield spans only when a whole partition worth of transcripts have been processed, then clear spans
        num_ts_processed = transcript_id + 1
        if num_ts_processed % num_ts_in_part == 0:
            yield spans
            spans = []


y1 = []
y2 = []
for part_id, spans_in_part in enumerate(gen_spans_by_partition(transcripts)):
    y1i = len(spans_in_part)
    y2i = len(set(spans_in_part)) / len(spans_in_part)
    y1.append(y1i)
    y2.append(y2i)

    print(f'Partition {part_id:>6,} | Found {y1i :>6,} {PATTERN_NAME} spans of which {y2i:>6,} are unique')


x1 = [i + 1 for i in range(len(y1))]
x2 = [i + 1 for i in range(len(y2))]

save_summary_to_txt(x=x1,
                    y=y1,
                    quantity_name=f'num_occurrences_of_{PATTERN_NAME}',
                    )

save_summary_to_txt(x=x1,
                    y=y2,
                    quantity_name=f'percent_unique_of_{PATTERN_NAME}',
                    )
