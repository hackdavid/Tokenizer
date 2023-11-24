from typing import List
import json
'''
# first download the data that are going to used in this whole process
# get dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
'''
class CharacterLevelTokenizerTranier:
  def __int__(self,file_path:str,speical_token:List=[],tokenizer_path:str='tokenizer'):
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
