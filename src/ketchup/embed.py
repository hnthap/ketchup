import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from .utils import flush


_model_id = 'sentence-transformers/all-mpnet-base-v2'
_tokenizer = AutoTokenizer.from_pretrained(_model_id)
_model = AutoModel.from_pretrained(_model_id)


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
    flush(verbose=False)
    return result


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
