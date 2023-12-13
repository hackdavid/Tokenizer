import sentencepiece as spm
from typing import Any, List

class CharacterLevelTokenizerTranier:
    def __init__(self,file_path:str,vocab_size:int,tokenizer_path:str='tokenizer',model_type:str='char',character_coverage=0.9995):

        '''
        we are not passing special_token because sentencepiece already using it internally so you dont have to worry 
        and also you can read more about sentencepiece for better understanding
        '''
        self.file_path = file_path
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage

    def train(self):
    
        print('Traning start............')

        spm.SentencePieceTrainer.train(f'--input={self.file_path} --model_prefix={self.tokenizer_path} --vocab_size={self.vocab_size} --model_type={self.model_type} \
                                    --character_coverage={self.character_coverage}')
        print('Traning is finished successfully...')
        print(f'Tokenizer is trained and save to file {self.tokenizer_path}')


class CharacterLevelTokenier:
    def __init__(self,tokenizer_file_path:str) -> None:
        self.tokenizer_file_path = tokenizer_file_path
    def tokenizer(self):
        return spm.SentencePieceProcessor(model_file=self.tokenizer_file_path)


  
  
