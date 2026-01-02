from typing import List, Optional
import os


class Tokenizer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._tokenizer = None
        self._loaded = False
        self._vocab_size = 151552
        self._eos_token_id = 151329
        self._pad_token_id = 151329

    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
            )
            self._vocab_size = self._tokenizer.vocab_size
            self._eos_token_id = self._tokenizer.eos_token_id or 151329
            self._pad_token_id = self._tokenizer.pad_token_id or self._eos_token_id
            self._loaded = True
            return True
        except Exception:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "THUDM/glm-4-9b",
                    trust_remote_code=True,
                )
                self._vocab_size = 151552
                self._eos_token_id = 151329
                self._pad_token_id = 151329
                self._loaded = True
                return True
            except Exception:
                self._loaded = False
                return False

    def is_loaded(self) -> bool:
        return self._loaded

    def encode(self, text: str) -> List[int]:
        if self._tokenizer:
            return self._tokenizer.encode(text, add_special_tokens=True)
        return self._simple_encode(text)

    def decode(self, token_ids: List[int]) -> str:
        if self._tokenizer:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return self._simple_decode(token_ids)

    def decode_single(self, token_id: int) -> str:
        if self._tokenizer:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)
        return self._simple_decode([token_id])

    def apply_chat_template(self, messages: List[dict]) -> str:
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def _simple_encode(self, text: str) -> List[int]:
        tokens = []
        for char in text:
            tokens.append(ord(char) % self._vocab_size)
        return tokens

    def _simple_decode(self, token_ids: List[int]) -> str:
        chars = []
        for tid in token_ids:
            if tid < 128:
                chars.append(chr(tid))
            else:
                chars.append(f"[{tid}]")
        return "".join(chars)


class MockTokenizer:
    def __init__(self, vocab_size: int = 151552):
        self._vocab_size = vocab_size
        self._eos_token_id = 151329

    def load(self) -> bool:
        return True

    def is_loaded(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        return [ord(c) % self._vocab_size for c in text]

    def decode(self, token_ids: List[int]) -> str:
        return "".join(chr(t % 128) if t < 128 else f"[{t}]" for t in token_ids)

    def decode_single(self, token_id: int) -> str:
        return chr(token_id % 128) if token_id < 128 else f"[{token_id}]"

    def apply_chat_template(self, messages: List[dict]) -> str:
        return "\n".join(m.get("content", "") for m in messages)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id
