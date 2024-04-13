from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self, rerank_model):
        self.model = CrossEncoder(rerank_model, max_length=512)

    def rerank(self, query, passages, batch_size):
        passage_texts = [(query, passage["title"] + " " + passage["text"]) for passage in passages]
        num_passages = len(passages)
        reranked_passages = []

        for i in range(0, num_passages, batch_size):
            batch = passage_texts[i:i + batch_size]
            scores = self.model.predict(batch).tolist()

            for j, passage in enumerate(passages[i:i + batch_size]):
                score_rerank = scores[j] if type(scores) is list else scores
                passage["score"] = score_rerank
                reranked_passages.append(passage)

        # Sort passages by scores in descending order
        reranked_passages.sort(key=lambda x: x["score"], reverse=True)
        return reranked_passages
