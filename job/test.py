'''
from transformers import BertTokenizerFast,AutoModel

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("ckiplab/gpt2-base-chinese")
s1 = "这是一个字"
s2 = "这是另一个字"
encoded_input = tokenizer(text=s1, text_pair=s2, return_tensors='pt', padding='max_length', max_length=90)
print(encoded_input)
output = model(**encoded_input)
#print(output.keys())
print(output.__dict__.keys())
'''
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModel
'''
tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
model = AutoModel.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
output = model(**encoded_input)
print(output.__dict__.keys())
'''
from transformers import AutoTokenizer, AutoModel

from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline, BartModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BartModel.from_pretrained("fnlp/bart-base-chinese")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input.keys())
output = model(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
print(output.__dict__.keys())
print(output.last_hidden_state)