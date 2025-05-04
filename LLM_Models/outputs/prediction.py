import os
import sys
import torch
import re
from django.shortcuts import render

# Add the parent directory to the path
sys.path.append(os.path.abspath(".."))

# Local imports
from LLM_Models.custom_tokenizer.tokenizer import get_tokenizer
from LLM_Models.model.transformer_block import Transformer
from LLM_Models.dataset.qa_dataset import QADataset

# Hyperparameters
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\Admin\Desktop\LLM_QA\training\data\processed\qa_transformer.pt"

# Tokenizer and model loading
tokenizer = get_tokenizer()
model = Transformer(vocab_size=tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Prediction function
def predict_answer(question, context):
    q_lower = question.strip().lower()
    stopwords = {"the", "in", "of", "a", "an", "is", "and", "water"}
    keyword = ""
    rephrased_question = question

    if "importance of" in q_lower:
        m = re.search(r"importance of ([\w\s]+?)(?: in aquaponics)?\??$", q_lower)
        if m:
            cand = m.group(1).strip()
            if cand and cand.lower() not in stopwords and len(cand.split()) == 1:
                keyword = cand
                rephrased_question = f"Why is {keyword} important in aquaponics?"
            else:
                return f'There is no defined “importance” for "{cand}" in aquaponics.'  # Return only a string
        else:
            return "No specific keyword found to answer that question."  # Return only a string

    # Tokenize input
    enc = tokenizer(
        rephrased_question,
        context,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attn_mask)

    # Compute probabilities
    start_probs = torch.softmax(start_logits.squeeze(0), dim=-1)
    end_probs = torch.softmax(end_logits.squeeze(0), dim=-1)
    scores = torch.matmul(start_probs.unsqueeze(1), end_probs.unsqueeze(0))

    seq_len = scores.size(0)
    mask = torch.triu(torch.ones((seq_len, seq_len), device=DEVICE))
    scores *= mask

    s_idx, e_idx = divmod(scores.argmax().item(), seq_len)
    tokens = input_ids[0, s_idx:e_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    if (not answer or answer.lower() in stopwords) and keyword:
        return f'There is no defined “importance” for "{keyword}" in aquaponics.'  # Return only a string

    if keyword and answer.lower().startswith(keyword.lower()):
        answer += (
            ". This is important because it enables sustainable agriculture, "
            "reduces environmental impact, and supports both plant and fish growth naturally."
        )

    if not answer:
        answer = "No answer could be generated for this question."

    return answer  # Return only the answer as a string
