from typing import Dict, List, Tuple
import numpy as np
from langchain_core.documents import Document


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def docs_to_dict(
    doc: List[Tuple[Document, float]], primary_key_field: str = "id"
) -> Tuple[Dict[str, Document], Dict[str, float], Dict[str, int]]:
    """Convert a list of documents and scores to dictionaries.

    Args:
        doc: List of (document, score) tuples

    Returns:
        Tuple of dictionaries containing document, score and rank information
    """
    doc_dict, score_dict, rank_dict = {}, {}, {}

    for i, (document, score) in enumerate(doc):
        uid_str = document.metadata.get(primary_key_field)
        # store the document, score and rank
        doc_dict[uid_str] = document
        score_dict[uid_str] = score
        rank_dict[uid_str] = i + 1

    return doc_dict, score_dict, rank_dict


def remove_duplicates(
    docs: List[tuple[Document, float]], primary_key_field: str = "id"
) -> List[tuple[Document, float]]:
    """Deduplicate documents based on their IDs.

    Args:
        docs: List of tuples containing (document, score)

    Returns:
        List of deduplicated documents preserving the original order
    """
    seen_ids = set()
    ret = []

    for doc, score in docs:
        id = doc.metadata.get(primary_key_field)
        if id not in seen_ids:
            seen_ids.add(id)
            ret.append((doc, score))

    return ret


def hybridCombine(
    keyword_docs: List[tuple[Document, float]],
    vector_docs: List[tuple[Document, float]],
    max_rank: int,
    ks_weight: float = 1.0,
    vs_weight: float = 1.0,
    rank_offset: int = 60,
    primary_key_field: str = "id",
) -> List[tuple[Document, float]]:
    """Combine keyword and vector retrieval using a hybrid reranking strategy.

    Args:
        keyword_docs: List of (document, score) tuples from keyword search
        vector_docs: List of (document, score) tuples from vector search
        max_rank: Maximum rank to consider
        ks_weight: Weight for keyword search scores (default: 1.0)
        vs_weight: Weight for vector search scores (default: 1.0)
        rank_offset: Offset added to rank to smooth scores (default: 60)

    Returns:
        List of reranked (document, score) tuples
    """
    all_docs = {}
    hybrid_scores = {}

    _, _, ks_rank_dict = docs_to_dict(keyword_docs, primary_key_field)
    _, _, vs_rank_dict = docs_to_dict(vector_docs, primary_key_field)

    for doc, _ in keyword_docs + vector_docs:
        id = doc.metadata.get(primary_key_field)
        if id not in all_docs:
            all_docs[id] = doc

            ks_rank = ks_rank_dict.get(id, max_rank + 1)
            vs_rank = vs_rank_dict.get(id, max_rank + 1)

            score = 0.0
            if ks_rank <= max_rank:
                score += ks_weight / (ks_rank + rank_offset)
            if vs_rank <= max_rank:
                score += vs_weight / (vs_rank + rank_offset)

            hybrid_scores[id] = score

    return sorted(
        ((doc, hybrid_scores[pid]) for pid, doc in all_docs.items()),
        key=lambda x: x[1],
        reverse=True,
    )


def get_features_to_use(keyword: bool, vector: bool, rerank: bool):
    """Get the list of features to use for the fusion model.

    Args:
        keyword: Whether keyword search is enabled
        vector: Whether vector search is enabled
        rerank: Whether reranking is enabled

    Returns:
        List of feature IDs to use
    """
    features = []
    if keyword:
        features.extend([1, 2, 3])
    if vector:
        features.extend([4, 5, 6])
    if rerank:
        features.extend([7, 8, 9])
    return features


def compute_document_features(
    keyword_docs: List[tuple[Document, float]],
    vector_docs: List[tuple[Document, float]],
    reranked_docs: List[tuple[Document, float]],
    primary_key_field: str = "id",
):
    """Compute document features for the fusion model.

    Args:
        keyword_docs: List of (document, score) tuples from keyword search
        vector_docs: List of (document, score) tuples from vector search
        reranked_docs: List of (document, score) tuples from reranker

    Returns:
        List of documents and their non-zero features
    """
    _, ks_score_dict, ks_rank_dict = docs_to_dict(keyword_docs, primary_key_field)
    _, vs_score_dict, vs_rank_dict = docs_to_dict(vector_docs, primary_key_field)
    reranked_docs_dict, reranked_score_dict, reranked_rank_dict = docs_to_dict(
        reranked_docs, primary_key_field
    )

    docs, doc_features = [], []
    for pid in reranked_docs_dict.keys():
        docs.append(reranked_docs_dict[pid])

        features = []
        features.append(0)  # placeholder
        features.append(ks_rank_dict.get(pid, -1))  # 1. keyword rank
        features.append(ks_score_dict.get(pid, 0))  # 2. keyword score
        miss = 1 if ks_rank_dict.get(pid, -1) == -1 else 0
        features.append(miss)  # 3. keyword miss

        features.append(vs_rank_dict.get(pid, -1))  # 4. vector rank
        features.append(vs_score_dict.get(pid, 0))  # 5. vector score
        miss = 1 if vs_rank_dict.get(pid, -1) == -1 else 0
        features.append(miss)  # 6. vector miss

        assert pid in reranked_rank_dict
        features.append(reranked_rank_dict[pid])  # 7. rerank rank
        features.append(reranked_score_dict[pid])  # 8. rerank score
        features.append(0)  # 9. placeholder
        doc_features.append(features)

    features_to_use = get_features_to_use(
        bool(keyword_docs), bool(vector_docs), bool(reranked_docs)
    )

    non_zero_features = []
    for i, data in enumerate(doc_features):
        features = []
        if features_to_use:
            for f_id in features_to_use:
                f_value = data[int(f_id)]
                if f_value != 0.0:
                    features.append(f"{f_id}:{f_value}")

        non_zero_features.append([str(data[0])] + features)

    return docs, non_zero_features
