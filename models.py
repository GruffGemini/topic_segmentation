import torch
from enum import Enum
from torch.nn.functional import pad
from transformers import BertModel, BertTokenizer, XLMTokenizer, XLMWithLMHeadModel


class CommonModelName(Enum):
    XLM = 'xlm'
    MBERT = 'mbert'


class XLMModel:
    name = 'xlm-mlm-xnli15-1024'
    PARALLEL_INFERENCE_INSTANCES = 1000

    def __init__(self, threshold: float):
        self.threshold = threshold

        # Initialize the tokenizer and model
        self.tokenizer = XLMTokenizer.from_pretrained(self.name)
        self.model = XLMWithLMHeadModel.from_pretrained(self.name)

        # Use a GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

    def get_features_from_sentence(self, batch_sentences, layer=-2):
        # Tokenize the input sentences
        input_ids = [self.tokenizer.encode(sentence, return_tensors='pt') for sentence in batch_sentences]

        # Pad the sequences with zeros to the maximum sequence length
        max_length = max([input_id.size(1) for input_id in input_ids])
        input_ids = [pad(input_id, pad=(0, max_length - input_id.size(1))) for input_id in input_ids]

        # Concatenate the padded sequences along the batch dimension
        input_ids = torch.cat(input_ids, dim=0)

        input_ids.to(self.device)

        # Extract the features from the model
        langs = torch.tensor([self.tokenizer.lang2id["en"]] * max_length)
        langs = langs.repeat(len(batch_sentences), 1)
        langs.to(self.device)

        with torch.no_grad():
            all_layers = self.model.forward(input_ids.to(self.device),
                                            attention_mask=input_ids.to(self.device).ne(0),
                                            output_hidden_states=True, langs=langs.to(self.device)
                                            )[-1]

        # Average the features of the specified layer across the tokens in each sentence
        layer_output = all_layers[layer]
        pooling = torch.nn.AvgPool2d((max_length, 1))
        batch_features = pooling(layer_output).squeeze()  # layer_output.mean(dim=1)
        if batch_features.dim() == 1:
            batch_features = batch_features.unsqueeze(dim=0)
        return batch_features


class MBertModel:
    name = 'bert-base-multilingual-cased'
    PARALLEL_INFERENCE_INSTANCES = 100

    def __init__(self, threshold: float):
        self.threshold = threshold

        # Initialize the tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.name)
        self.model = BertModel.from_pretrained(self.name)

        # Use a GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

    def get_features_from_sentence(self, batch_sentences, layer=-2):
        # Tokenize the input sentences
        input_ids = [self.tokenizer.encode(sentence, return_tensors='pt') for sentence in batch_sentences]

        # Pad the sequences with zeros to the maximum sequence length
        max_length = max([input_id.size(1) for input_id in input_ids])
        input_ids = [pad(input_id, pad=(0, max_length - input_id.size(1))) for input_id in input_ids]

        # Concatenate the padded sequences along the batch dimension
        input_ids = torch.cat(input_ids, dim=0)

        input_ids.to(self.device)

        # Extract the features from the model
        with torch.no_grad():
            all_layers = self.model.forward(input_ids.to(self.device),
                                            attention_mask=input_ids.to(self.device).ne(0),
                                            output_hidden_states=True
                                            )[-1]

        # Average the features of the specified layer across the tokens in each sentence
        layer_output = all_layers[layer]
        pooling = torch.nn.AvgPool2d((max_length, 1))
        batch_features = pooling(layer_output).squeeze()  # layer_output.mean(dim=1)
        if batch_features.dim() == 1:
            batch_features = batch_features.unsqueeze(dim=0)
        return batch_features
