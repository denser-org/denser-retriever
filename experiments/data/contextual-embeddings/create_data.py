import json
from langchain_core.documents import Document
import os
import pickle


def create_contextual_data(original_data_dir, output_data_dir, add_anthropic_context):
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    out_passages = open(os.path.join(output_data_dir, 'passages.jsonl'), 'w')
    out_queries = open(os.path.join(output_data_dir, 'queries.jsonl'), 'w')
    out_qrels = open(os.path.join(output_data_dir, 'qrels.jsonl'), 'w')

    if add_anthropic_context:
        with open(os.path.join(output_data_dir, "contextual_vector_db.pkl"), "rb") as file:
            data = pickle.load(file)
        meta = data["metadata"]

    with open(os.path.join(original_data_dir, 'codebase_chunks.json'), 'r') as input_file:
        doc_id = 0
        docs = json.loads(input_file.read())
        for doc in docs:
            doc_uuid = doc["original_uuid"]
            for chunk in doc['chunks']:
                if add_anthropic_context:
                    page_content = chunk.pop('content') + "\n\n" + meta[doc_id]['contextualized_content']
                    doc_id += 1
                else:
                    page_content = chunk.pop('content')
                metadata = chunk
                metadata['pid'] = doc_uuid + "_" + str(metadata['original_index'])
                new_doc = Document(page_content=page_content, metadata=metadata)
                out_passages.write(json.dumps(new_doc.dict(), ensure_ascii=False) + "\n")

        if add_anthropic_context:
            assert doc_id == len(meta)

    with open(os.path.join(original_data_dir, 'evaluation_set.jsonl'), 'r') as input_file:
        query_id = 0
        for line in input_file:
            data = json.loads(line)
            query_dict = {"id": str(query_id), "text": data['query']}
            out_queries.write(json.dumps(query_dict) + '\n')
            labels = []
            for gold_doc, passage_index in data['golden_chunk_uuids']:
                labels.append(gold_doc + "_" + str(passage_index))
            query_to_labels = {str(query_id): {label: 1 for label in labels}}
            out_qrels.write(json.dumps(query_to_labels) + '\n')
            query_id += 1


if __name__ == "__main__":
    original_data_dir = "experiments/data/contextual-embeddings/original_data"

    output_data_dir = "experiments/data/contextual-embeddings/data_base"
    add_anthropic_context = False
    create_contextual_data(original_data_dir, output_data_dir, add_anthropic_context)

    output_data_dir = "experiments/data/contextual-embeddings/data_context"
    add_anthropic_context = True
    create_contextual_data(original_data_dir, output_data_dir, add_anthropic_context)
