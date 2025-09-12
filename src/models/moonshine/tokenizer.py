""" CharacterTokenzier for Hugging Face Transformers.

This is heavily inspired from CanineTokenizer in transformers package.
"""
import json
import os
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers import AutoTokenizer
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


__all__ = [
    "CharacterTokenizer",
    "TeluguTokenizer",
    "extract_telugu_tokens_from_pretrained"
]

class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[BOS]": 0
                    "[EOS]": 1
                    "[PAD]": 2
                    "[UNK]": 3
                an id (starting at 4) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        self._vocab_str_to_int = {
            "[BOS]": 0,
            "[EOS]": 1,
            "[PAD]": 2,
            "[UNK]": 3,
            **{ch: i + 4 for i, ch in enumerate(characters)},
        }
        # print(self._vocab_str_to_int)
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        # print(self._vocab_int_to_str)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
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
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

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
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

# chars = string.ascii_lowercase + " !," # This is character vocab
# print(chars)
# model_max_length = 2048
# tokenizer = CharacterTokenizer(chars, model_max_length)

# example = ["i love nlp!", "nlp is great!"]
# tokens = tokenizer.batch_encode_plus(
#     example,
#     add_special_tokens=True,
#     # max_length=model_max_length,
#     padding="longest",
#     # truncation=True,
#     return_tensors="tf",
# )
# example = "i love nlp!@"
# tokens = tokenizer(example)
# print(tokens)

# print(tokenizer.decode(tokens["input_ids"]))
# print(tokenizer.batch_decode(tokens["input_ids"]))
# print(tokenizer.vocab_size)

class TeluguTokenizer(PreTrainedTokenizer):
    def __init__(self, telugu_tokens: Sequence[str], model_max_length: int, **kwargs):
        """Telugu tokenizer for Hugging Face transformers.

        Args:
            telugu_tokens (Sequence[str]): List of Telugu tokens extracted from a pretrained model.
                Any token which is not included in this list will be replaced by a special token 
                called [UNK] with id=3. Following are list of all special tokens with their ids:
                    "[BOS]": 0
                    "[EOS]": 1
                    "[PAD]": 2
                    "[UNK]": 3
                An id (starting at 4) will be assigned to each Telugu token.

            model_max_length (int): Model maximum sequence length.
        """
        self.telugu_tokens = telugu_tokens
        self.model_max_length = model_max_length
        
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        self._vocab_str_to_int = {
            "[BOS]": 0,
            "[EOS]": 1,
            "[PAD]": 2,
            "[UNK]": 3,
            **{token: i + 4 for i, token in enumerate(telugu_tokens)},
        }
        
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        
        self._sorted_tokens = sorted(telugu_tokens, key=len, reverse=True)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
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
    """Extract Telugu tokens from a pretrained tokenizer with deterministic ordering.
    
    Args:
        model_name (str): Name or path of the pretrained model
        
    Returns:
        List[str]: List of Telugu tokens found in the vocabulary
    """
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