import regex as re
import base64
import os
import json
import tiktoken
from transformers import PreTrainedTokenizer
from typing import List, Optional, Union, Dict
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding


class ChatGLM4Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
            self,
            vocab_file,
            padding_side="left",
            clean_up_tokenization_spaces=False,
            encode_special_tokens=False,
            **kwargs
    ):
        self.name = "GLMTokenizer"
        self.vocab_file = vocab_file
        pat_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self.pat_str = re.compile(pat_str)
        self.encode_special_tokens = encode_special_tokens

        mergeable_ranks = {}
        with open(vocab_file) as f:
            for line in f:
                token, rank = line.strip().split()
                rank = int(rank)
                token = base64.b64decode(token)
                mergeable_ranks[token] = rank

        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = ["<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>",
                               "<|user|>", "<|assistant|>", "<|observation|>", "<|begin_of_image|>", "<|end_of_image|>",
                               "<|begin_of_video|>", "<|end_of_video|>"]

        self.special_tokens = {
            token: idx for idx, token in enumerate(self.special_tokens, start=len(mergeable_ranks))
        }
        self.special_token_ids = {idx: token for token, idx in self.special_tokens.items()}

        self.tokenizer = tiktoken.Encoding(
            name="my_tokenizer",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )
        self.decoder = {rank: token for token, rank in mergeable_ranks.items()}
        self.n_words = len(self.decoder) + len(self.special_tokens)

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    def get_command(self, token):
        assert token in self.special_tokens
        return self.special_tokens[token]

    @property
    def vocab_size(self):
        return self.n_words

    @property
    def eos_token_id(self):
        return self.get_command("<|endoftext|>")

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors="replace")
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors="replace")
        return text

    def _tokenize(self, text, **kwargs):
        tokens = []
        if self.encode_special_tokens:
            ids = self.tokenizer.encode(text, allowed_special="all")
        else:
            ids = self.tokenizer.encode(text, disallowed_special=())
        for t in ids:
            tokens.append(self.decoder[t])
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.mergeable_ranks[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_token_ids:
            return self.special_token_ids[index]
        return self.decoder[index]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("<sop>")]
        return prefix_tokens

    def build_single_message(self, role, metadata, message):
        assert role in ["system", "user", "assistant", "observation"], role
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message, disallowed_special=())
        tokens = role_tokens + message_tokens
        return tokens

    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                for function in item["tools"]:
                    content += f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
                    content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
