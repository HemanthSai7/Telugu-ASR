import json
import os
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import tensorflow as tf
from transformers import AutoTokenizer
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


__all__ = [
    "TeluguTokenizer",
    "extract_telugu_tokens_from_pretrained"
]

class TeluguTokenizer(PreTrainedTokenizer):
    def __init__(self, telugu_tokens: Sequence[str], model_max_length: int, **kwargs):
        self.telugu_tokens = telugu_tokens
        self.model_max_length = model_max_length
        
        blank_token = AddedToken("[BLANK]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        self._vocab_str_to_int = {
            "[BLANK]": 0,
            "[UNK]": 1,
            **{token: i + 2 for i, token in enumerate(telugu_tokens)},
        }
        
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        self._sorted_tokens = sorted(telugu_tokens, key=len, reverse=True)

        self.blank_token_id = self._vocab_str_to_int["[BLANK]"]

        super().__init__(
            blank_token=blank_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def get_vocab(self):
        return self._vocab_str_to_int

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Telugu text using longest-first matching."""
        tokens = []
        i = 0
        
        while i < len(text):
            # Try to match the longest possible token first
            matched = False
            for token in self._sorted_tokens:
                if text[i:i+len(token)] == token:
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            
            if not matched:
                # Add UNK token instead of raw character
                tokens.append("[UNK]")
                i += 1
        
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    
    def prepend_blank(self, token_ids: List[int]) -> List[int]:
        print("############", [self.blank_token_id] + token_ids)
        return tf.concat([[self.blank_token_id], token_ids], axis=0)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        eos = [self.eos_token_id]
        bos = [self.bos_token_id]
        result = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            result += token_ids_1 + eos
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "telugu_tokens": self.telugu_tokens,
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "TeluguTokenizer":
        return cls(
            telugu_tokens=config["telugu_tokens"],
            model_max_length=config["model_max_length"]
        )


def extract_telugu_tokens_from_pretrained(model_name: str) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    
    # Telugu Unicode block range
    telugu_start = 0x0C00
    telugu_end = 0x0C7F
    
    # Use a set for deduplication but preserve the extraction order
    seen_tokens = set()
    telugu_tokens = []
    
    # Sort by token_id to ensure deterministic order
    for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        if token in seen_tokens:
            continue
            
        is_telugu = False
        
        # Check if token contains Telugu characters by Unicode range
        if any(telugu_start <= ord(char) <= telugu_end for char in token):
            is_telugu = True
        else:
            for char in token:
                try:
                    name = unicodedata.name(char)
                    if "TELUGU" in name:
                        is_telugu = True
                        break
                except ValueError:
                    pass
        
        if is_telugu:
            telugu_tokens.append(token)
            seen_tokens.add(token)
    
    # Sort by length for better tokenization (longest first)
    telugu_tokens.sort(key=len, reverse=True)
    
    print(f"Extracted {len(telugu_tokens)} Telugu tokens from {model_name}")
    return telugu_tokens


# model_name = "google/gemma-2-2b"
# telugu_tokens = extract_telugu_tokens_from_pretrained(model_name)
# telugu_tokens.append(" ")
# tokenizer = TeluguTokenizer(telugu_tokens, model_max_length=1204)
# ids = tokenizer.encode("కరెంటు బిల్లుల్ని కూడా సులువుగా ఆండ్రాయిడ్ ఫోన్ ఉంటే జీపే ద్వారా చెయ్యొచ్చు* ", add_special_tokens=False)
# ids_with_blank = tokenizer.prepend_blank(ids)
# text = tokenizer.decode(ids_with_blank, skip_special_tokens=True)
# text = text.replace("[BLANK]", "")
# print(ids_with_blank, text)