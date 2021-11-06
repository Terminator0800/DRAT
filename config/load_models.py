'''
Created on 2021年4月27日

@author: Administrator
'''
def load_pretrained_embeddings():
    all_pretrained_embedding_map = {}
    lines = list(open("../../data/pretrained_models/other/pretrained_char_vec.txt", 'r', encoding='utf8').readlines())
    for line in lines:
        slices = line.strip().split(" ")
        token, vector = slices[0], slices[1:]
        vector = list(map(lambda x: float(x), vector))
        all_pretrained_embedding_map[token] = vector
        
    all_tokens = list(open("../../data/pretrained_models/chinese_L-12_H-768_A-12/vocab.txt", 'r', encoding='utf8').readlines())
    pretrained_embedding = [None for i in range(len(all_tokens))]
    for i in range(len(all_tokens)):
        token = all_tokens[i].strip()
        if token in all_pretrained_embedding_map:
            pretrained_embedding[i] = all_pretrained_embedding_map[token]
        else:
            pretrained_embedding[i] = [0 for i in range(100)]
    
    return pretrained_embedding

if __name__ == '__main__':
    load_pretrained_embeddings()
            