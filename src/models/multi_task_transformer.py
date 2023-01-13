import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartForSequenceClassification, BartForConditionalGeneration


class BartForEventDetectionAndSummarization(nn.Module):
    def __init__(self, cls_config: BartConfig, summ_config: BartConfig, *args, **kwargs):
        self.cls_model = BartForSequenceClassification(cls_config)
        self.summ_model = BartForConditionalGeneration(summ_config)
        # TODO - config or from_pretrained??

        # share embeddings, encoder and decoder
        self.cls_model.model.shared = self.summ_model.model.shared
        self.cls_model.model.encoder = self.summ_model.model.encoder
        self.cls_model.model.decoder = self.summ_model.model.decoder

        # what is not shared is the classification head
        # cls_model has the BartClassificationHead
        # summ_model has lm_head (which does language modelling)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"If you want to do classification, use `forward_cls`. For summarization, use `forward_summ`.")

    def forward_cls(self, *args, **kwargs):
        return self.cls_model.forward(*args, **kwargs)

    def forward_summ(self, *args, **kwargs):
        return self.summ_model.forward(*args, **kwargs)

