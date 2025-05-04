from transformers import BertTokenizerFast

def get_tokenizer(model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return tokenizer
