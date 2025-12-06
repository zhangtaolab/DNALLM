import os
import json
import torch
import torch.nn as nn


class DNAOneHotTokenizer:
    def __init__(
        self,
        max_length: int | None = 196_608,
        padding_side: str = "right",
        return_embeds: bool = False,
        embeds_transpose: bool = False,
    ):
        self.max_length = max_length
        self.padding_side = padding_side
        self.return_embeds = return_embeds
        self.embeds_transpose = embeds_transpose

        self.token_to_id = {
            "A": 0,
            "a": 0,
            "C": 1,
            "c": 1,
            "G": 2,
            "g": 2,
            "T": 3,
            "t": 3,
            "U": 3,
            "u": 3,  # RNA
            "N": 4,
            "n": 4,
            "X": 4,
            "x": 4,
            "-": -1,
            ".": -1,  # Padding
        }

        self.id_to_token = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N", -1: "-"}

        self.pad_token_id = -1
        self.pad_token = "-"  # noqa: S105
        self.unk_token_id = 4  # N

        self.vocab_vectors = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0: A
                [0.0, 1.0, 0.0, 0.0],  # 1: C
                [0.0, 0.0, 1.0, 0.0],  # 2: G
                [0.0, 0.0, 0.0, 1.0],  # 3: T / U
                [0.25, 0.25, 0.25, 0.25],  # 4: N / X
                [0.0, 0.0, 0.0, 0.0],  # 5: Padding, index=-1
            ],
            dtype=torch.float32,
        )

    def __call__(
        self,
        sequences: str | list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        return_dict: bool = True,
        return_input_ids: bool = True,
        return_inputs_embeds: bool | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor] | torch.Tensor | None:
        if isinstance(sequences, str):
            sequences = [sequences]

        max_len = max_length if max_length is not None else self.max_length

        batch_ids = []
        for seq in sequences:
            ids = self.convert_tokens_to_ids(list(seq))
            batch_ids.append(ids)

        if padding and max_len:
            target_len = max_len
        else:
            target_len = max(len(ids) for ids in batch_ids)

        padded_ids = []
        attention_masks = []
        pad_id = self.pad_token_id

        for ids in batch_ids:
            current_len = len(ids)

            if truncation and max_len and current_len > max_len:
                ids = ids[:max_len]
                current_len = max_len

            pad_len = target_len - current_len
            if pad_len > 0:
                padding_tokens = [pad_id] * pad_len
                if self.padding_side == "right":
                    ids = ids + padding_tokens
                    mask = [1] * current_len + [0] * pad_len
                else:
                    ids = padding_tokens + ids
                    mask = [0] * pad_len + [1] * current_len
            else:
                mask = [1] * current_len

            padded_ids.append(ids)
            attention_masks.append(mask)

        ids_tensor = torch.tensor(padded_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)

        output = {}
        if return_inputs_embeds is None:
            return_inputs_embeds = self.return_embeds
        if return_dict:
            output["attention_mask"] = attention_mask

        if return_input_ids:
            if return_dict:
                output["input_ids"] = ids_tensor
            elif not return_inputs_embeds:
                return ids_tensor

        if return_inputs_embeds:
            idx = ids_tensor.clone()
            idx[idx == -1] = 5  # Padding index
            one_hot_output = nn.functional.embedding(idx, self.vocab_vectors)
            if self.embeds_transpose:
                one_hot_output = one_hot_output.transpose(1, 2)

            if return_dict:
                output["inputs_embeds"] = one_hot_output
            else:
                return one_hot_output

        if return_tensors == "np" and return_dict:
            for key in output:
                output[key] = output[key].numpy()

        return output if return_dict else None

    def convert_tokens_to_ids(
        self, tokens: list[str] | str
    ) -> list[int] | int:
        if isinstance(tokens, str):
            return self.token_to_id.get(tokens, self.unk_token_id)
        return [
            self.token_to_id.get(token, self.unk_token_id) for token in tokens
        ]

    def convert_ids_to_tokens(self, ids: list[int] | int) -> list[str] | str:
        if isinstance(ids, int):
            return self.id_to_token.get(ids, "N")
        return [self.id_to_token.get(i, "N") for i in ids]

    def encode(
        self, text: str, max_length: int | None = None, truncation: bool = True
    ) -> list[int]:
        ids = self.convert_tokens_to_ids(list(text))
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for i in token_ids:
            if skip_special_tokens and i == self.pad_token_id:
                continue
            tokens.append(self.id_to_token.get(i, "N"))

        return "".join(tokens)

    def batch_decode(
        self,
        sequences: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]

    @property
    def vocab_size(self):
        return 6

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save tokenizer configuration and vocabulary.
        Generates: tokenizer_config.json and vocab.json
        """
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "max_length": self.max_length,
            "padding_side": self.padding_side,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "tokenizer_class": self.__class__.__name__,
        }

        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

        return [config_file, vocab_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load Tokenizer from a specified directory.
        """
        config_file = os.path.join(
            pretrained_model_name_or_path, "tokenizer_config.json"
        )

        if os.path.isfile(config_file):
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # Remove some metadata that should not be passed to __init__
            config.pop("tokenizer_class", None)
            config.pop("pad_token_id", None)
            config.pop("unk_token_id", None)

            config.update(kwargs)

            return cls(**config)
        else:
            print(
                "Warning: tokenizer_config.json not found",
                f"in {pretrained_model_name_or_path}. Using default init.",
            )
            return cls(**kwargs)
