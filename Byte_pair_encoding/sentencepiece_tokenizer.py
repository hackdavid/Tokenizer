import sentencepiece as spm
from typing import Any, List

class BPETokenizerTranier:
    def __init__(self,file_path:str,vocab_size:int,tokenizer_path:str='sp_tokenizer',model_type:str='bpe',character_coverage=0.9995):

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
                                    --character_coverage={self.character_coverage} --split_digits=true \
                                        --remove_extra_whitespaces=false')
        print('Traning is finished successfully...')
        print(f'Tokenizer is trained and save to file {self.tokenizer_path}')


class BPETokenier:
    def __init__(self,tokenizer_file_path:str) -> None:
        self.tokenizer_file_path = tokenizer_file_path
    def tokenizer(self):
        # we dont have to implement encode and decode method because sentencepiece provide this method for encoding and decoding
        return spm.SentencePieceProcessor(model_file=self.tokenizer_file_path)
    
if __name__ == '__main__':
    text = 'input.txt'
    trainer = BPETokenizerTranier(
        file_path=text,
        vocab_size=1000
    )

    trainer.train()
    model_path = 'sp_tokenizer.model'
    bpe = BPETokenier(tokenizer_file_path=model_path)
    tokenizer = bpe.tokenizer()
    print(tokenizer)
    sample = ''' hello my name is
    david from nepal
    '''
    encode = tokenizer.encode(sample)
    print(f'encode token : {encode}')
    print(f'Decode token : {tokenizer.decode(encode)}')

