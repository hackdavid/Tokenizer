from typing import List
import json
'''
# first download the data that are going to used in this whole process
# get dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
'''
class CharacterLevelTokenizerTranier:
    def __init__(self,file_path:str,speical_token:List=[],tokenizer_path:str='tokenizer'):
        self.file_path = file_path
        self.special_token = speical_token
        self.tokenizer_path = f'{tokenizer_path}.json'
        self.vocab = {}
    def train(self):
        f = open(self.file_path,'r')
        all_text = f.readlines()
        index = 0
        # adding special token first
        if self.special_token:
            for token in self.special_token:
                if token not in self.vocab:
                    self.vocab.update({token:index})
                    index += 1
        # adding all character in lower_case
        for each_line in all_text:
            for chr in each_line:
                chr = chr.lower()
                if chr not in self.vocab:
                    self.vocab.update({chr:index})
                    index += 1
        # saving vocab in .json file 
        with open(self.tokenizer_path,'w') as f:
            data = json.dumps(self.vocab)
            f.write(data)
        print(f'Tokenizer is trained and save to file {self.tokenizer_path}')



class CharacterLevelTokenier:
    def __init__(self,tokenizer_file_path:str) -> None:
        self.tokenizer_file_path = tokenizer_file_path
        # load the json file and make an object so we can use it for enoding and decoding
        with open(self.tokenizer_file_path,'r') as f:
            data = f.read()
            self.tokenizer = json.loads(data)
  
    def encode(self,prompt:str):
        '''
        Convert the prompt into array of int 
        '''
        sentences = prompt.split('.') # breaking into sentence so we can add eso token
        response = []
        for sent in sentences:
            for chr in sent:
                chr = chr.lower()
                response.append(self.tokenizer.get(chr,self.tokenizer['<unk>'])) # if character is not found then replace that character with unknow token id
            response.append(self.tokenizer.get('</s>'))
        return response

    def decode(self,token:List):
        response = []
        for ids in token:
            response.append(self.tokenizer.get(ids))
        return ''.join(response)


