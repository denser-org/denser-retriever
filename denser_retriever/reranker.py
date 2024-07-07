import copy

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, rerank_model: str):
        self.model = CrossEncoder(rerank_model, trust_remote_code=True, max_length=512)

    def rerank(self, query, passages, batch_size, query_id=None):
        passages_copy = copy.deepcopy(passages)
        passage_texts = [
            (query, passage["title"] + " " + passage["text"])
            for passage in passages_copy
        ]
        num_passages = len(passages_copy)
        reranked_passages = []

        for i in range(0, num_passages, batch_size):
            batch = passage_texts[i : i + batch_size]
            scores = self.model.predict(batch, batch_size=batch_size, convert_to_tensor=True).tolist()

            for j, passage in enumerate(passages_copy[i : i + batch_size]):
                score_rerank = scores[j] if isinstance(scores, list) else scores
                passage["score"] = score_rerank
                reranked_passages.append(passage)

        # Sort passages by scores in descending order
        reranked_passages.sort(key=lambda x: x["score"], reverse=True)
        return reranked_passages
