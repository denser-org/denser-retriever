from langchain_core.documents import Document

import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_HF_corpus_as_docs(corpus, output_file: str, max_doc_size, max_doc_len):
    out = open(output_file, "w")
    seen = set()
    for i, d in enumerate(corpus):
        if max_doc_size > 0 and i >= max_doc_size:
            break
        page_content = d.pop("text")
        if max_doc_len > 0 and len(page_content) > max_doc_len:
            page_content = page_content[:max_doc_len]
        d["pid"] = d.pop("id")
        assert d["pid"] not in seen
        seen.add(d["pid"])
        doc = Document(page_content=page_content, metadata=d)
        json.dump(doc.dict(), out, ensure_ascii=False)
        out.write("\n")


def save_data(
        group_data, output_feature, output_group, features
):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data)) + "\n")

    for i, data in enumerate(group_data):
        # only include nonzero features
        feats = []

        for p in data[2:]:
            f_name, f_value = p.split(":")
            if features and int(f_name) not in features:
                continue
            if float(f_value) != 0.0:
                feats.append(p)

        output_feature.write(data[0] + " " + " ".join(feats) + "\n")


def prepare_features(
        exp_dir, out_file, out_group_file, features_to_use
):
    fi = open(os.path.join(exp_dir, "features.svmlight"))
    output_feature = open(os.path.join(exp_dir, out_file), "w")
    output_group = open(os.path.join(exp_dir, out_group_file), "w")

    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line:
            line = line[: line.index("#")]
        splits = line.strip().split(" ")
        if splits[1] != group:
            # print(f"Processing group {group}")
            save_data(
                group_data,
                output_feature,
                output_group,
                features_to_use
            )
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(
        group_data, output_feature, output_group, features_to_use
    )

    fi.close()
    output_feature.close()
    output_group.close()
