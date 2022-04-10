"""
Find all spans in AO-CHILDES where a probe is preceded by a token tagged by spacy as "compound".

Save results to a txt file for plotting in Latex

"""
import spacy
import numpy as np
from spacy.tokens import Doc
from spacy.matcher import Matcher
from typing import List, Generator
from collections import defaultdict, Counter

from aochildes.dataset import AOChildesDataSet
from aochildespatterns.utils import save_summary_to_txt
from aochildespatterns.probes import probes

NUM_PARTS = 2
PATTERN_NAME = 'compound+target'
VERBOSE = False

transcripts_ = AOChildesDataSet().load_transcripts()

# make num transcripts divisible by NUM_PARTS
transcripts = transcripts_

nlp = spacy.load("en_core_web_sm", exclude=['ner'])

matcher = Matcher(nlp.vocab)

pattern = [{'DEP': 'compound', 'OP': "+"},  # adjectival modifier,
           {"TEXT": {"IN": probes}},
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
y3 = []
part_id2compounds = defaultdict(list)
compound2spans = defaultdict(list)
for part_id, spans_in_part in enumerate(gen_spans_by_partition(transcripts)):
    y1i = len(spans_in_part)
    y2i = len(set(spans_in_part))
    y3i = len(set(spans_in_part)) / len(spans_in_part)
    y1.append(y1i)
    y2.append(y2i)
    y3.append(y3i)

    print(f'Partition {part_id:>6,} | Found {y1i :>6,} {PATTERN_NAME} spans of which {y2i:>6,} are unique')

    # collect all compounds to see which become more frequent with age
    for span in spans_in_part:
        compound = span.split()[-2]
        part_id2compounds[part_id].append(compound)
        compound2spans[compound].append((part_id, span))


# which compound has greatest percent increase from part 1 to part 2?
if NUM_PARTS == 2:
    c0 = Counter(part_id2compounds[0])
    c1 = Counter(part_id2compounds[1])

    compound2pi = {}
    for compound in c1:
        f0 = c0.get(compound, 1)  # pretend each compound in part 2 is seen at least once in part 1
        f1 = c1[compound]
        fd = f1 - f0
        percent_increase = fd / f0
        print(f'{compound:<16} f0={f0:>6,} f1={f1:>6,} fd={fd:>6,} pi={percent_increase:.3f}')
        compound2pi[compound] = percent_increase

    for compound, pi in sorted(compound2pi.items(), key=lambda i: i[1])[-10:]:
        print(f'{compound:<16} pi={pi:.4f}')
        print(compound2spans[compound])

# summaries
save_summary_to_txt(x=[i + 1 for i in range(len(y1))],
                    y=y1,
                    quantity_name=f'num_occurrences_of_{PATTERN_NAME}',
                    )
save_summary_to_txt(x=[i + 1 for i in range(len(y2))],
                    y=y2,
                    quantity_name=f'num_unique_of_{PATTERN_NAME}',
                    )

save_summary_to_txt(x=[i + 1 for i in range(len(y3))],
                    y=y3,
                    quantity_name=f'percent_unique_of_{PATTERN_NAME}',
                    )


# make co-mat, one for each partition, then compute fragmentation for each
for part_id in range(NUM_PARTS):

    # init matrix
    num_cols = len(compound2spans)
    num_rows = len(probes)
    co_mat = np.zeros((num_rows, num_cols))
    left_contexts = list(compound2spans.keys())
    # collect co-occurrences
    for left_context, spans in compound2spans.items():
        col_id = left_contexts.index(left_context)
        for part_id_, span in spans:
            if part_id_ == part_id:
                probe = span.split()[-1]
                row_id = probes.index(probe)
                co_mat[row_id, col_id] += 1

    # compute fragmentation
    u, s, vt = np.linalg.svd(co_mat, compute_uv=True)
    assert np.max(s) == s[0]
    frag = 1 - (s[0] / np.sum(s))
    print(f'partition={part_id:>4} frag = {frag}')
