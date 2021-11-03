"""
Find all spans in AO-CHILDES where a probe is preceded by a token tagged by spacy as "amod".

Save results to a txt file for plotting in Latex

"""
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from typing import List, Generator
from collections import defaultdict, Counter

from aochildes.dataset import ChildesDataSet
from aochildespatterns.utils import save_summary_to_txt
from aochildespatterns.probes import probes

NUM_PARTS = 2
PATTERN_NAME = 'amod+PROBE'
VERBOSE = False

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
            if VERBOSE:
                print(span.text)
            spans.append(span.text)

        # yield spans only when a whole partition worth of transcripts have been processed, then clear spans
        num_ts_processed = transcript_id + 1
        if num_ts_processed % num_ts_in_part == 0:
            yield spans
            spans = []


y1 = []
y2 = []
part_id2amods = defaultdict(list)
amod2spans = defaultdict(list)
for part_id, spans_in_part in enumerate(gen_spans_by_partition(transcripts)):
    y1i = len(spans_in_part)
    y2i = len(set(spans_in_part)) / len(spans_in_part)
    y1.append(y1i)
    y2.append(y2i)

    print(f'Partition {part_id:>6,} | Found {y1i :>6,} {PATTERN_NAME} spans of which {y2i:>6,} are unique')

    # collect all amods to see which become more frequent with age
    for span in spans_in_part:
        amod = span.split()[-2]
        part_id2amods[part_id].append(amod)
        amod2spans[amod].append((part_id, span))


# which amod has greatest percent increase from part 1 to part 2?
if NUM_PARTS == 2:
    c0 = Counter(part_id2amods[0])
    c1 = Counter(part_id2amods[1])

    amod2pi = {}
    for amod in c1:
        f0 = c0.get(amod, 1)  # pretend each amod in part 2 is seen at least once in part 1
        f1 = c1[amod]
        fd = f1 - f0
        percent_increase = fd / f0
        print(f'{amod:<16} f0={f0:>6,} f1={f1:>6,} fd={fd:>6,} pi={percent_increase:.3f}')
        amod2pi[amod] = percent_increase

    for amod, pi in sorted(amod2pi.items(), key=lambda i: i[1])[-10:]:
        print(f'{amod:<16} pi={pi:.4f}')
        print(amod2spans[amod])

# summaries
save_summary_to_txt(x=[i + 1 for i in range(len(y1))],
                    y=y1,
                    quantity_name=f'num_occurrences_of_{PATTERN_NAME}',
                    )
save_summary_to_txt(x=[i + 1 for i in range(len(y2))],
                    y=y2,
                    quantity_name=f'percent_unique_of_{PATTERN_NAME}',
                    )