import json
import pickle

import faiss
import dotenv
import logging
from pathlib import Path
from munch import DefaultMunch
from main import summarize_query, embed_text, search_top_k_with_scores


query = "找老婆的标准是什么？"
# query = "cs学生如何规划未来？"
# query = "兔子和猪哪个更聪明？"
config_dict = {
    "paths": {
        "vector_index": "./knowledge/faiss_index.bin",
        "processed_track": "./knowledge/processed_files.json"
    },
    "models": {
        "query_summarization": "ep-20250212105505-5zlbx",  # doubao 1.5 pro
        "text_embedding": "doubao-embedding-text-240715"
    },
    "qa": {
        "top_k": 10
    },
    "generative": {
        "max_summary_tokens": 4096
    }
}
config = DefaultMunch.fromDict(config_dict)


if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path='../.env', override=True)

    vector_index_path = Path(config.paths.vector_index)
    raw = faiss.read_index(str(vector_index_path))
    idx = faiss.IndexIDMap(raw) if not isinstance(raw, faiss.IndexIDMap) else raw
    print(f"Knowledge base loaded")

    short_q = summarize_query(query, config)
    print(f"Short question: {short_q}\n")
    q_emb = embed_text(short_q, config)

    processed_track_path = Path(config.paths.processed_track)
    processed = json.loads(processed_track_path.read_text(encoding="utf-8")) \
        if processed_track_path.exists() else {}
    int2hex = {
        int(int(hex_id, 16) % (10 ** 12)): hex_id
        for hex_id in processed.keys()
    }

    res = search_top_k_with_scores(idx, q_emb, config.qa.top_k)
    topk_doc_int_ids, sims = zip(*res)
    topk_doc_hex_ids = [int2hex[int_id] for int_id in topk_doc_int_ids]
    topk_doc_paths = [processed[hex_id]['dst_file_path'] for hex_id in topk_doc_hex_ids]
    topk_src_doc_paths = [processed[hex_id]['src_file_path'] for hex_id in topk_doc_hex_ids]

    for path, src_path, sim in zip(topk_doc_paths, topk_src_doc_paths, sims):
        print(f"({round(sim, 4)}) {path}\n\t{src_path}")

        with open(path, "rb") as f:
            d = pickle.load(f)
            # print(f"\t{d["summary"]}\n{d["original"]}\n\n")
            print(f"\t{d["summary"]}\n")
