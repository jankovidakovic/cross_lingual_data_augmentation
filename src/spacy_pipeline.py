from functools import partial
from itertools import islice
from typing import Iterable, Generator

from spacy import Language


def process(*,
            nlp: Language,
            texts: Iterable[str],
            n_process: int,
            n_sents: int | None = None,
            **kwargs
            ) -> Generator[str, None, None]:
    with nlp.select_pipes(
            enable=[
                "transformer",
                "tagger",
                "attribute_ruler",
                "sentencizer",
                "lemmatizer"
            ]
    ):
        for doc in nlp.pipe(texts, n_process=n_process):
            # generate sentences
            sents = (sent for sent in doc.sents)

            # cutoff sentences, if any
            if n_sents:
                sents = islice(sents, 0, n_sents)

            # yield lemmas
            yield (token.lemma_ for sent in sents for token in sent)


def make_pipeline(
        nlp: Language,
        n_process: int,
        n_sents: int | None = None,
        **kwargs
):
    return partial(
        process,
        nlp=nlp,
        n_process=n_process,
        n_sents=n_sents,
        **kwargs
    )
