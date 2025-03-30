import gc
import torch

from sentence_transformers import SentenceTransformer
import spacy


_model = SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2',
    device='cuda',
)
_nlp = spacy.load('en_core_web_trf')
_nlp_disabled_components = _nlp.pipe_names
_nlp.add_pipe('sentencizer')


def get_embeddings(sentences: list[str]) -> list[list[float]]:
    return _model.encode(sentences)


def sentencize(text: str, *, nlp: spacy.language.Language=_nlp):
    return list(map(
        lambda s: s.text.strip(),
        nlp(text, disable=_nlp_disabled_components).sents,
    ))


def sentencize_batch(
        texts: list[str], 
        *, 
        nlp: spacy.language.Language=_nlp,
):
    return list(map(
        lambda result: list(map(
            lambda s: s.text.strip(),
            result.sents,
        )),
        nlp.pipe(texts, disable=_nlp_disabled_components),
    ))


def flush(*, verbose=True):
    if verbose:
        print('Garbage collector flushed %d objects' % gc.collect())
    else:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
