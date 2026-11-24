from typing import Any, Dict, List

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_chatglm2 import (
    ChatGLMTokenizer as ChatGLMV2TokenizerHf,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_chatglm3 import (
    ChatGLMTokenizer as ChatGLMV3TokenizerHf,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_chatglm4 import (
    ChatGLM4Tokenizer as ChatGLMV4TokenizerHf,
)


class ChatGLMV2Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = ChatGLMV2TokenizerHf.from_pretrained(tokenizer_path)


class ChatGLMV3Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = ChatGLMV3TokenizerHf.from_pretrained(
            tokenizer_path, encode_special_tokens=True
        )


class ChatGLMV4Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = ChatGLMV4TokenizerHf.from_pretrained(tokenizer_path)


class ChatGLMV5Tokenizer(BaseTokenizer):
    """
    GLM-5 specific tokenizer to handle invalid tokenizer_class in config.

    Problem: GLM-5's tokenizer_config.json contains "tokenizer_class": "TokenizersBackend"
             which doesn't exist in transformers library, and has format issues with
             extra_special_tokens that causes AttributeError.

    Solution:
    1. Load tokenizer.json directly using tokenizers library
    2. Wrap it with PreTrainedTokenizerFast manually
    3. Avoid reading tokenizer_config.json entirely

    This works because:
    - tokenizer.json contains all vocabulary and encoding rules
    - We bypass all config parsing that causes issues
    - No dependency on the problematic tokenizer_config.json

    Building the tokenizer from a tokenizer object also skips two things that
    live outside tokenizer.json and that generation needs, so both are restored
    below: the chat template (its own ``chat_template.jinja`` file in current
    transformers) and the full stop-token set from ``generation_config.json``.
    Without them the model answers, emits ``<|user|>`` and keeps going with a
    turn it made up, because ``<|user|>`` is a stop token nothing registered.
    """

    #: ``eos_token`` below covers this one; the rest become stop words.
    PRIMARY_EOS = "<|endoftext|>"

    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        import os

        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        # Load tokenizer directly from tokenizer.json, bypassing tokenizer_config.json
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer = Tokenizer.from_file(tokenizer_file)

        # Wrap it with PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token=self.PRIMARY_EOS,
            pad_token=self.PRIMARY_EOS,
            unk_token=self.PRIMARY_EOS,
        )
        self._load_chat_template(tokenizer_path)
        self._extra_stop_token_ids = self._load_extra_stop_token_ids(tokenizer_path)

    def _load_chat_template(self, tokenizer_path: str) -> None:
        import os

        # Transformers moved the template out of tokenizer_config.json into a
        # sibling file; GLM-5.3-Flash ships it that way. ``renderer_factory``
        # only picks a template up when the tokenizer carries one.
        template_file = os.path.join(tokenizer_path, "chat_template.jinja")
        if not os.path.exists(template_file):
            return
        with open(template_file, encoding="utf-8") as handle:
            self.tokenizer.chat_template = handle.read()

    def _load_extra_stop_token_ids(self, tokenizer_path: str) -> List[int]:
        import json
        import logging
        import os

        config_file = os.path.join(tokenizer_path, "generation_config.json")
        if not os.path.exists(config_file):
            return []
        with open(config_file, encoding="utf-8") as handle:
            eos = json.load(handle).get("eos_token_id")
        if eos is None:
            return []
        if isinstance(eos, int):
            eos = [eos]
        primary = self.tokenizer.convert_tokens_to_ids(self.PRIMARY_EOS)
        extra = [int(token_id) for token_id in eos if int(token_id) != primary]
        if extra:
            logging.info(
                "GLM-5 stop tokens beyond %s (id %s): %s",
                self.PRIMARY_EOS,
                primary,
                [(i, self.tokenizer.convert_ids_to_tokens(i)) for i in extra],
            )
        return extra

    @property
    def stop_words_id_list(self) -> List[List[int]]:
        return [[token_id] for token_id in getattr(self, "_extra_stop_token_ids", [])]


register_tokenizer(["chatglm2", "chat_glm_2"], ChatGLMV2Tokenizer)
register_tokenizer(["chatglm3", "chat_glm_3"], ChatGLMV3Tokenizer)
register_tokenizer(["chatglm4", "chatglm4v"], ChatGLMV4Tokenizer)
register_tokenizer(["glm_5", "glm5_next", "glm5_next_mtp"], ChatGLMV5Tokenizer)
