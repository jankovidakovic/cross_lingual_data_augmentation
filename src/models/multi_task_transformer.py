import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartModel


class MultiTaskTransformer(nn.Module):
    def __init__(self, config: BartConfig, *args, **kwargs):
        super().__init__()
        self.transformer = BartModel(config)
        self.encoder: BartEncoder = self.transformer.get_encoder()
        self.decoder: BartDecoder = self.transformer.get_decoder()
        self.cls_head = None
        self.language_modelling_head = nn.Linear(
            config.d_model, self.transformer.shared.num_embeddings,
            bias=False
        )  # why no bias here?

    def forward(self):
        raise NotImplementedError(
            f"forward is not implemented for {self.__class__.__name__}."
            "Use either `forward_sequence_classification` or `forward_language_modelling`"
        )

    def forward_sequence_classification(self, input):
        # forward encoder
        # forward linear classification head
        # return logits
        pass

    def forward_language_modelling(self, input):
        # forward whole model
        # forward LM head
        # return logits
        pass

