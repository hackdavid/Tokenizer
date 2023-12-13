In this section, i am going to show you how to train , use your own custom BPE tokenizer.

this section contains
  1. implement BPE algorithm
  2. use huggingface library to train and use tokenizer with our custom dataset
  3. use sentencepiece library to train and use tokenizer with our custom dataset

so lets take a look into it one by one.
# 1 implement BPE alogrithm
  just open bpq_alogrithm.ipynb and you will get each line explaination involved<br>
  <a href='bpe_algorithm.ipynb'>open notebook</a>

# 2 use huggingface librabry
  hugginface can be used to train custom tokenizer for different different dataset or langauge.
  lets talk a real example where you need to train your own tokenizer.
      suppose if you want to train or fine-tune a pre-trained model(i will show you on this series later) for different langauge like hindi,telgu,marathi but pre-trained model 
      does not support or tokenize in proper way on this langauge so in this case you have to prepare dataset for target language
      like telgu or marathi and then build a tokenizer using this dataset .

  I hope you will understand how it is important to know how to train your custom tokenizer.<br>
  <a href='huggingface_tokenizer.py'>view code </a> visit the source code and just go inside the main function and pass the dataset path
  and you will get .json file which contains all the token details.
