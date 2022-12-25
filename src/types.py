from typing import TypeVar
from dataclasses import dataclass


from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel


# those types ensure statis type checking to work
#   subsequently, autocomplete works better
#   altho it doesnt make sense because those are runtime values,
#   which cannot be statically autocompleted
ModelConfigType = TypeVar("ModelConfigType", bound=PretrainedConfig)
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizer)
ModelType = TypeVar("ModelType", bound=PreTrainedModel)


@dataclass
class NewsWikiSplit:
    import pandas as pd
    from dataclasses import field
    from typing import Callable

    news: pd.DataFrame = field(kw_only=True)
    wiki: pd.DataFrame = field(kw_only=True)

    def map(self, f: Callable[[pd.DataFrame], pd.DataFrame]):
        return NewsWikiSplit(news=f(self.news), wiki=f(self.wiki))
