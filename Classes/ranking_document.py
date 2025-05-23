import os
import json
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_dense(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])


def run_spade_experiment(city, model_name="sentence-transformers/all-mpnet-base-v2", output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning SPaDE experiment for: {city}")
    
    dataset_path = os.path.join("Dataset/qa/", "questionpassages_group_city.json")
    output_path = os.path.join(output_dir, f"{city}_ranking.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[city]

    passage_ids = list(data.keys())
    passages = [data[pid]['passage'] for pid in passage_ids]
    questions = [data[pid]['question'] for pid in passage_ids]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    dense_passage_embeddings = encode_dense(passages, tokenizer, model)
    dense_question_embeddings = encode_dense(questions, tokenizer, model)

    vectorizer = TfidfVectorizer().fit(passages + questions)
    sparse_passage_embeddings = vectorizer.transform(passages)
    sparse_question_embeddings = vectorizer.transform(questions)

    rankings = {}
    for i, pid in tqdm(enumerate(passage_ids), total=len(passage_ids), desc="SPaDE Ranking"):
        dense_scores = torch.nn.functional.cosine_similarity(
            dense_question_embeddings[i].unsqueeze(0), dense_passage_embeddings
        ).numpy()

        sparse_scores = cosine_similarity(
            sparse_question_embeddings[i], sparse_passage_embeddings
        ).flatten()

        combined_scores = 0.5 * dense_scores + 0.5 * sparse_scores
        top_indices = np.argsort(combined_scores)[::-1]

        ranking = [{
            "rank": rank + 1,
            "passage_id": passage_ids[idx],
            "score": float(combined_scores[idx]),
            "passage": passages[idx]
        } for rank, idx in enumerate(top_indices)]

        rankings[pid] = {"question": data[pid]['question'], "ranking": ranking}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"SPaDE-style rankings saved to '{output_path}'")


def run_bm25_experiment(city, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning BM25 experiment for: {city}")

    dataset_path = os.path.join("Dataset/qa/", "questionpassages_group_city.json")
    output_path = os.path.join(output_dir, f"{city}_ranking.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[city]

    passage_ids = list(data.keys())
    passages = [data[pid]['passage'] for pid in passage_ids]
    questions = [data[pid]['question'] for pid in passage_ids]

    tokenized_passages = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)

    rankings = {}
    for i, pid in tqdm(enumerate(passage_ids), total=len(passage_ids), desc="BM25 Ranking"):
        query_tokens = questions[i].lower().split()
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)

        ranking = [{
            "rank": rank + 1,
            "passage_id": passage_ids[idx],
            "score": float(scores[idx]),
            "passage": passages[idx]
        } for rank, idx in enumerate(top_indices)]

        rankings[pid] = {"question": questions[i], "ranking": ranking}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"BM25 rankings saved to '{output_path}'")


def run_dense_cosine_experiment(city, model_name='all-mpnet-base-v2', output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning dense cosine experiment for: {city}")

    dataset_path = os.path.join("Dataset/qa/", "questionpassages_group_city.json")
    output_path = os.path.join(output_dir, f"{city}_ranking.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[city]

    passage_ids = list(data.keys())
    passages = [data[pid]['passage'] for pid in passage_ids]
    questions = [data[pid]['question'] for pid in passage_ids]

    model = SentenceTransformer(model_name)
    passage_embeddings = model.encode(passages, convert_to_tensor=True, show_progress_bar=True)
    query_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)

    rankings = {}
    cos_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)

    for i, pid in enumerate(passage_ids):
        scores = cos_scores[i]
        top_scores, top_indices = torch.topk(scores, k=len(passage_ids), largest=True)

        ranking = [{
            "rank": rank + 1,
            "passage_id": passage_ids[idx],
            "score": float(top_scores[rank]),
            "passage": passages[idx]
        } for rank, idx in enumerate(top_indices)]

        rankings[pid] = {"question": data[pid]['question'], "ranking": ranking}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"Dense cosine rankings saved to '{output_path}'")