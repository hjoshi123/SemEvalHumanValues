from typing import List, Dict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_Tokenizer():
    return tokenizer
    
def tokenize_and_encode(row):
    return tokenizer(row['Premise'] + ' ' 
                     + row['Stance'] + ' '
                     + row['Conclusion'], truncation=True)

def detokenize_and_decode(tokened_arguments):
    decoded_tokens = []
    for i in range(len(tokened_arguments)):
        decoded_tokens.append(tokenizer.hf_tokenizer.batch_decode(tokened_arguments[i]["input_ids"]))
    return decoded_tokens