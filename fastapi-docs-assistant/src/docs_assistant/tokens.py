"""Token counting with the same tokenizer the embedding model uses.

Chunk limits are in tokens, not characters: models bill and truncate by
tokens, and code tokenizes much more densely than English prose.
"""
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken
    return tiktoken.get_encoding(config.TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))
