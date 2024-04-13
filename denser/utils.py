from typing import List, Dict, Union, Tuple
import torch
import json
import logging
import sys


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    MRR = {}
    
    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"]/len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR

def recall_cap(qrels: Dict[str, Dict[str, int]], 
               results: Dict[str, Dict[str, float]], 
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    capped_recall = {}
    
    for k in k_values:
        capped_recall[f"R_cap@{k}"] = 0.0
    
    k_max = max(k_values)
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"] += (len(retrieved_docs) / denominator)

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"]/len(qrels), 5)
        logging.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    return capped_recall


def hole(qrels: Dict[str, Dict[str, int]], 
               results: Dict[str, Dict[str, float]], 
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    Hole = {}
    
    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0
    
    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():    
            annotated_corpus.add(doc_id)
    
    k_max = max(k_values)
    logging.info("\n")
    
    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"]/len(qrels), 5)
        logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole

def top_k_accuracy(
        qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    top_k_acc = {}
    
    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"]/len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc

loggers = {}


def get_logger(name='default'):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s: %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger
        return logger

def get_logger_file(name):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=name,
                        filemode='a')
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    return logging

def save_denser_corpus(corpus, output_file: str, max_doc_size):
    out = open(output_file, "w")
    for i, d in enumerate(corpus):
        if max_doc_size > 0 and i >= max_doc_size:
            break
        data = {"source": d['id'],
                "title": d["title"],
                "text": d["text"],
                "pid": -1}
        json.dump(data, out, ensure_ascii=False)
        out.write('\n')


def save_denser_queries(queries, output_file: str):
    out = open(output_file, "w")
    for d in queries:
        data = {"id": d['id'],
                "text": d["text"]
                }
        json.dump(data, out, ensure_ascii=False)
        out.write('\n')


def save_denser_qrels(qrels, output_file: str):
    out = open(output_file, "w")
    for q in qrels.keys():
        data = {q: qrels[q]}
        json.dump(data, out, ensure_ascii=False)
        out.write('\n')
def dump_passages(passages: List[Dict[str, str]], output_file: str):
    out = open(output_file, "w")
    for passage in passages:
        json.dump(passage, out, ensure_ascii=False)
        out.write('\n')

def passages_to_dict(passages: List[Dict[str, str]], doc_task, score_name = "score"):
    res = {}
    for passage in passages:
        source, pid = passage["source"], passage.get("pid", -1)
        if doc_task or pid == -1:
            uid_str = source
        else:
            uid_str = f"{source}-{pid}"
        assert uid_str not in res
        res[uid_str] = passage[score_name]
    return res

def aggregate_passages(passages: List[Dict[str, str]]):
    """
    Using the highest score among the passages as the document's score. Each passage represents a unique doc.
    """
    # import pdb; pdb.set_trace()
    uid_to_passages, uid_to_scores = {}, {}
    for i, passage in enumerate(passages):
        uid_str = passage["source"]
        if uid_str not in uid_to_passages:
            uid_to_passages[uid_str] = passage
            uid_to_scores[uid_str] = passage["score"]
        elif uid_to_scores[uid_str] < passage["score"]:
            uid_to_passages[uid_str] = passage
            uid_to_scores[uid_str] = passage["score"]

    docs =  list(uid_to_passages.values())
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs