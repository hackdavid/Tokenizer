import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

class CharacterLevelTokenizerTranier(PreTrainedTokenizer):
    def __init__(self,file_path:str,vocab_size:int,tokenizer_path:str='hf_tokenizer',**kwargs):
        self.file_path = file_path
        self.tokenizer_prefix = tokenizer_path
        self.vocab_size = vocab_size
        # adding special characters
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=self.vocab_size,
            **kwargs,
        )
    def train(self):
        print(f'Traning tokenizer ........')
        # get all character first
        self.get_character()
        # make mapping for char-to-int and also int-to-char
        # as we know special token is added first so we have to make this as first tokens
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.character)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        # saving tokenizer config
        self.save_pretrained()
        print(f'Traning is complete successfully and save config to {self.tokenizer_prefix}')

    def get_character(self):
        self.character = []
        f = open(self.file_path,'r')
        all_text = f.readlines()
        for each_line in all_text:
            for ch in each_line:
                if ch not in self.character:
                    self.character.append(ch)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    
    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.character],
            "model_max_length": self.vocab_size,
        }

    def save_pretrained(self):
        cfg_file = f"{self.tokenizer_prefix}_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)



class CharacterLevelTokenizer(PreTrainedTokenizer):
    '''
    No need to define decode and encode method for this becuase huggingface pretrainedTokenizer class provide this method
    '''
    @classmethod
    def from_pretrained(cls,tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
    


        

    
        
            