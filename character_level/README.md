# 1. Character Level Tokenizer

*   first i will implement character level tokenizer alogrithm 
*   train our tokenizer using custome dataset using our own alogrithm
*   train our tokenizer with sentencepiece libray
*   train our tokenizer with huggingface 
*.  we see the difference tokenizer train by different different approches as mention above

first download the data that are going to used in this whole process

        !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

lets implement the character level alogrithm

         import json
         def character_level_tokenizer_alogrithm(file_path:str,special_token=[],output_path:str='tokenizer.json'):
                  # special_token - read below to know more about special_token
                  # in character level simply we store only unique character with socre/index(integer) which will use for decoding 
                  
                  # read the file 
                  text = open(file_path,'r')
                  text_with_lines = text.readlines()
                  # we are going assign space also with a index and \n will also including
                  all_chars = {}
                  indx = 0
                  print(indx)
                  # adding special token at first 
                  if special_token:
                    for token in special_token:
                      all_chars.update({token:indx})
                      indx += 1
                  for each_line in text_with_lines:
                    for i in each_line:
                      if i.lower() not in all_chars:
                        # make sure all character in lower case
                        all_chars.update({i.lower():indx})
                        indx += 1
                  # save this dict into .json file (you can store in any format ,i am using .json because hugginface store into .json but sentencepiece store in .vocab)
                  with open(output_path,'w') as f:
                    data = json.dumps(all_chars)
                    f.write(data)
                  print(f'Tokenizer is save with file_name {output_path}')
                  return all_chars


Note: if you are famliar with tokenizer then there are some specail which all kept inside vocab so we have to also add that special token and all special token indx/score will start from 0 becuase we add these special token at starting

lets see the special token(maybe i miss some special token but you can add by yourself)

1. sos(start of sentence)
2. eos(end of sentence)
3. padding
4. unknow token

        '''
        sos = <s>
        eos = </s>
        padding = <padding>
        ukown  = <unk>
        '''
        special_token = ['<s>','</s>','<unk>','<padding>']
        vocab = vocab = character_level_tokenizer_alogrithm(dataset_path,special_token)

So this is simple alogrithm and used most of the NLP task becuase it takse less space and cover all the word by using these characters becase these are the base characters of any langauge .

lets build a tokenizer traniner so that can we use further for traning our custome tokenzier
which code is inside the train.py and you can aslo see the notebook attached to this directory for step by step implementation
