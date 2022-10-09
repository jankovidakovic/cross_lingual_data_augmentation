from typing import TypeVar

from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel


# those types ensure statis type checking to work
#   subsequently, autocomplete works better
#   altho it doesnt make sense because those are runtime values,
#   which cannot be statically autocompleted
ModelConfigType = TypeVar("ModelConfigType", bound=PretrainedConfig)
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizer)
ModelType = TypeVar("ModelType", bound=PreTrainedModel)
