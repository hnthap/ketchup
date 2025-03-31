import gc

import torch
import torch.nn.functional as F

import spacy
from transformers import AutoTokenizer, AutoModel


_model_id = 'sentence-transformers/all-mpnet-base-v2'
_tokenizer = AutoTokenizer.from_pretrained(_model_id)
_model = AutoModel.from_pretrained(_model_id)

_nlp = spacy.load('en_core_web_trf')
_nlp_disabled_components = _nlp.pipe_names
_nlp.add_pipe('sentencizer')


def set_sentence_transformer_device(device):
    _model.to(device)


def get_embeddings(sentences: list[str]) -> list[list[float]]:
    '''
    Encode a list of sentences into embeddings.
    Args:
        sentences (list[str]): A list of sentences.
    Returns:
        (list[list[float]]): A list of embeddings.
    '''
    encoded_input = _tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    encoded_input = {
        key: value.to(_model.device) for key, value in encoded_input.items()
    }
    with torch.no_grad():
        model_output = _model(**encoded_input)
    sentence_embeddings = _mean_pooling(
        model_output,
        encoded_input['attention_mask'],
    )
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    result = sentence_embeddings.cpu().detach().numpy().tolist()
    del encoded_input, model_output, sentence_embeddings
    torch.cuda.empty_cache()
    return result


def sentencize(text: str, *, nlp: spacy.language.Language=_nlp) -> list[str]:
    '''
    Split a text into sentences (i.e. to sentencize).
    Args:
        text (str): The input text.
        nlp (spacy.language.Language): The spaCy language model for
            tokenization. Default to a pre-defined model.
    Returns:
        (list[str]): A list of sentences.
    '''
    return list(map(
        lambda s: s.text.strip(),
        nlp(text, disable=_nlp_disabled_components).sents,
    ))


def sentencize_batch(
        texts: list[str], 
        *, 
        nlp: spacy.language.Language=_nlp,
) -> list[list[str]]:
    '''
    Split a batch of text into lists of sentences corresponding to each
    sentence (i.e. to sentencize).
    Args:
        texts (list[str]): A batch of text.
        nlp (spacy.language.Language): The spaCy language model for
            tokenization. Default to a pre-defined model.
    Returns:
        (list[list[str]]): Lists of sentences corresponding to each sentence.
    '''
    return list(map(
        lambda result: list(map(
            lambda s: s.text.strip(),
            result.sents,
        )),
        nlp.pipe(texts, disable=_nlp_disabled_components),
    ))


def flush(*, verbose=True):
    '''
    Collects all unused resources.
    Args:
        verbose (bool): Whether to print the number of objects collected by
            built-in garbage collector.
    '''
    if verbose:
        print('Garbage collector flushed %d objects' % gc.collect())
    else:
        gc.collect()
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def _mean_pooling(model_output, attention_mask):
    '''
    Mean pooling when encoding text with the sentence transformer model.
    Args:
        model_output (torch.Tensor): The output of the model.
        attention_mask (torch.Tensor): The attention mask.
    '''
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return (
        torch.sum(token_embeddings * input_mask_expanded, 1) /
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    )
