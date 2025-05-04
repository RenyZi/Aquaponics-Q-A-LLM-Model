import sys
import os
from fuzzywuzzy import fuzz
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from LLM_Models.dataset.qa_dataset import QADataset
from LLM_Models.outputs.prediction import predict_answer
from LLM_Models.model.feed_forward import FeedForward
from LLM_Models.model.encoder_layer import EncoderLayer
from LLM_Models.model.transformer_block import Transformer
from LLM_Models.custom_tokenizer.tokenizer import get_tokenizer
from LLM_Models.model.attetion import ScaledDotProductAttention
from LLM_Models.model.position_encoding import PositionEncoding
from LLM_Models.model.multihead_attetion import MultiHeadAttetion

# Ensure project path is set (optional redundancy)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def keyword_overlap_score(q, text):
    q_words = set(q.lower().split())
    text_words = set(text.lower().split())
    return len(q_words & text_words)

@csrf_exempt
def output(request):
    tokenizer = get_tokenizer()
    DATASET_PATH = r"C:\Users\Admin\Desktop\LLM_QA\data\processed\processed_aquaponics_dataset.json"
    dataset = QADataset(DATASET_PATH, tokenizer)

    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_question = request.POST.get("question", "").strip()

        if user_question:
            best_match = None
            best_score = 0

            for item in dataset:
                question_score = fuzz.token_set_ratio(user_question, item["question"])
                context_score = fuzz.partial_ratio(user_question, item["context"])
                overlap_score = keyword_overlap_score(user_question, item["context"] + " " + item["question"])
                combined_score = 0.5 * max(question_score, context_score) + 0.5 * overlap_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = item

            if best_match and best_score >= 30:
                raw_answer = predict_answer(user_question, best_match["context"])
                answer = raw_answer.strip().capitalize()
                if not answer.endswith('.'):
                    answer += '.'
            else:
                answer = f"Sorry, I couldn’t find a relevant context for your question: '{user_question}'."

            return JsonResponse({"answer": answer})

    # Only render the page on GET request
    return render(request, 'output.html')
