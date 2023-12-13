import sentencepiece as spm
from typing import Any, List

# load huggingface tokeizer class 
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizerTranier:
    def __init__(self,file_path:str,vocab_size:int,tokenizer_path:str='hf_tokenizer'):

        '''
        we are not passing special_token because sentencepiece already using it internally so you dont have to worry 
        and also you can read more about sentencepiece for better understanding
        '''
        self.file_path = file_path
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    def train(self):
    
        print('Traning start............')
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        self.tokenizer.pre_tokenizer = Whitespace()
        files = [self.file_path]
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(f"{self.tokenizer_path}")
        print('Traning is finished successfully...')
        print(f'Tokenizer is trained and save to file {self.tokenizer_path}')


class BPETokenier:
    def __init__(self,tokenizer_file_path:str) -> None:
        self.tokenizer_file_path = tokenizer_file_path
    def tokenizer(self):
        # we dont have to implement encode and decode method because sentencepiece provide this method for encoding and decoding
        return Tokenizer.from_file(self.tokenizer_file_path)


if __name__ == '__main__':
    text = 'input.txt'
    trainer = BPETokenizerTranier(
        file_path=text,
        vocab_size=1000
    )

    trainer.train()
    tokenizer_path = 'hf_tokenizer.json'
    bpe = BPETokenier(tokenizer_file_path=tokenizer_path)
    tokenizer = bpe.tokenizer()
    sample = ''' hello my name is
    david from nepal
    '''
    encode = tokenizer.encode(sample)
    print(f'encode token : {encode}')
    print(f'Decode token : {tokenizer.decode(encode.ids)}')

'''
Some encoding and decoding methods 

1. tokenizer.tokenize()
2. tokenizer.encode(text)
3. tokenizer.encode_batch([setnence1,sentence2,sentence3,......])
4. tokenizer.decode(Union[list:int,tupple:int])
5. tokenizer.decode_batch(list of list of ini)


'''