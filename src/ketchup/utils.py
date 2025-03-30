import gc

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import spacy


_tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-mpnet-base-v2'
)
_model = AutoModel.from_pretrained(
    'sentence-transformers/all-mpnet-base-v2',
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

_nlp = spacy.load('en_core_web_trf')
_nlp_disabled_components = _nlp.pipe_names
_nlp.add_pipe('sentencizer')


def get_embeddings(sentences: list[str]) -> list[list[float]]:
    encoded_input = _tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        model_output = _model(**encoded_input)
    sentence_embeddings = _mean_pooling(
        model_output,
        encoded_input['attention_mask'],
    )
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()


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


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return (
        torch.sum(token_embeddings * input_mask_expanded, 1) /
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    )
