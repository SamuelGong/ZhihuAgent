#!/usr/bin/env python3
import os
import sys
import time
import json
import math
import faiss
import dotenv
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from datetime import datetime
from collections import Counter
from rich.console import Console
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from zhihuagent.misc import load_config, set_log
from zhihuagent.model_related import get_model_response, get_embedding


def summarize_query(query, config):
    """Coarse-grain the user query for better embedding."""
    user_query = """
请将这句可能很长、很复杂的提问精炼到足够用于语义搜索的简短版本（如果足够剪短，可以保留原样）：

{query}   
"""
    user_query = user_query.format(query=query)
    resp, _ = get_model_response(
        model=config.models.query_summarization,
        messages=[
            {"role":"system", "content": "你是一个擅长将用户提问归纳成紧凑语句的助手。"},
            {"role":"user", "content": user_query}
        ],
        temperature=0.0,
        max_tokens=config.generative.max_summary_tokens
    )
    return resp.strip()


def embed_text(text: str, config) -> np.ndarray:
    emb = get_embedding(text=text, model=config.models.text_embedding)
    return np.array(emb, dtype="float32")


def search_top_k_with_scores(idx, q_emb: np.ndarray, k: int):
    """
    Returns a list of (int_id, score) sorted by best match first.
    For an IP index, score is the inner product (higher = better);
    for an L2 index, score is the squared L2 distance (lower = better).
    """
    D, I = idx.search(q_emb.reshape(1, -1), k)
    ids    = I[0].tolist()
    scores = D[0].tolist()

    # filter out any “-1” padding
    results = [
        (int(_id), _score)
        for _id, _score in zip(ids, scores)
        if _id != -1
    ]
    return results


def generate_answer_for_doc(doc_path, query, config, log_file):
    set_log(filename=log_file)
    with open(doc_path, "rb") as fin:
        doc_data = pickle.load(fin)
        doc_text = doc_data["original"]
        doc_summary = doc_data["summary"]

#     system_prompt = """
# 你是一个基于检索增强的问答助手。
# 在回答问题时，你必须基于提供的参考文档来支持每一个事实性陈述。
# 不要凭空捏造或虚构文档中没有的事实。
# 如果文档中不包含答案，请回答：“对不起，我不知道。”
# 否则，先在文中查找可能的答案证据，并在回答中注明出处原文（如：‘文档第2段中提到，…’），然后给出详细的答案。
# """

    system_prompt = """你是一个问答助手，参考指定文档给出回答。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"原始问题：{query}\n\n"
            f"【参考文档开始】\n\n{doc_text}\n\n【参考文档结束】\n\n"
            "请基于参考文档回答原始问题："
         }
    ]
    resp, _ = get_model_response(
        model=config.models.dialog,
        messages=messages,
        temperature=0.0,
        # max_tokens=config.generative.max_answer_tokens
    )
    resp = resp.strip()
    logging.info(f"Per-doc answering with {doc_path} "
                 f"(length: {len(resp)}):\n{resp[:500]}\n"
                 f"The summary of the original document is:\n{doc_summary[:500]}")
    return resp


def _chunk_list(lst, size):
    it = iter(lst)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def _summarize_chunk_worker(answers_chunk, query, config):
    # just forward to your existing hierarchical_summarize
    return hierarchical_summarize(query, answers_chunk, config)


def recursive_summarize(query: str, answers: list, config, level: int = 1) -> str:
    """
    Recursively summarize `answers` in parallel, chunking at each level,
    without any globals—arguments only.
    """
    chunk_size = config.qa.summarize_chunk_size
    max_depth  = config.qa.max_summarization_depth

    # Base case
    if len(answers) <= chunk_size or level >= max_depth:
        logging.info(f"[Summ L{level}] final pass on {len(answers)} answers")
        return hierarchical_summarize(query, answers, config)

    # Split into chunks
    chunks = list(_chunk_list(answers, chunk_size))
    logging.info(f"[Summ L{level}] {len(answers)} answers → {len(chunks)} chunks of ≤{chunk_size}")

    # Parallel summarization of each chunk
    intermediate = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        # map each chunk to a worker along with the same query & config
        futures = executor.map(
            _summarize_chunk_worker,
            chunks,
            [query]*len(chunks),
            [config]*len(chunks)
        )
        for summary in tqdm(
            futures,
            total=len(chunks),
            desc=f"Summ L{level}",
            leave=False
        ):
            intermediate.append(summary)

    # Recurse to next level
    return recursive_summarize(query, intermediate, config, level + 1)


def hierarchical_summarize(query, answers, config, feature=False):
    """Merge multiple answers into one deep insight."""
    if feature:
        system_prompt = """
你是一个擅长用一针见血的一句话回答问题的助手。
具体地，在回答用户的特定问题时，你会先根据多个专家的独立意见，形成你自己的自主看法。
然后，你不会设法简单机械地用一句话概括所有内容。
而是，你会将你觉得最重要的一点，用一句有文采或者有深度的话写出来（二三十字左右）。
"""
    else:
        system_prompt = """
你是一个擅长综合多方信息的问答助手。
在回答用户的特定问题时，你会基于下面的专家回答（它们均已引用文档），而不引入新的信息或幻觉。
但你不需要显式引用某个专家的回答，只需要用你的话陈述即可。
你只会使用自然语言段落来和人类友好互动，而从来不使用 Markdown 或特殊符号。 
"""

    joined = "\n\n---\n\n".join(f"回答{i+1}：{a}" for i, a in enumerate(answers))
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content":
            f"原始问题：{query}\n\n"
            "以下是多个专家的独立意见：\n\n"
            f"{joined}\n\n"
            "请使用自然语言段落，将它们整合成一个信息量丰富、有具体细节的最终回答（不要使用 Markdown 格式）："
        }
    ]
    resp, _ = get_model_response(
        model=config.models.answer_summarization,
        messages=messages,
        temperature=0.2,
    )
    resp = resp.strip()

    logging.info(f"Final answer:\n{resp}")
    return resp


def main():
    # 1) bootstrap
    begin_time = time.perf_counter()

    dotenv.load_dotenv(dotenv_path='.env', override=True)
    console = Console()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"qa_{timestamp}.log"
    set_log(filename=str(log_file))
    config = load_config(os.getcwd())
    logging.info(f"Config\n{json.dumps(config, indent=4)}")

    # 2) load indices & mappings
    processed_track_path = Path(config.paths.processed_track)
    processed = json.loads(processed_track_path.read_text(encoding="utf-8")) \
        if processed_track_path.exists() else {}
    int2hex = {
        int(int(hex_id, 16) % (10 ** 12)): hex_id
        for hex_id in processed.keys()
    }

    inverted_index_path = Path(config.paths.inverted_index)
    inverted = json.loads(inverted_index_path.read_text(encoding="utf-8")) \
        if inverted_index_path.exists() else {}

    vector_index_path = Path(config.paths.vector_index)
    raw = faiss.read_index(str(vector_index_path))
    idx = faiss.IndexIDMap(raw) if not isinstance(raw, faiss.IndexIDMap) else raw
    logging.info(f"Knowledge base loaded")

    # 3) greet & read first query
    console.print(":mag_right: [bold yellow]欢迎使用知乎知识库问答助手！[/]")
    console.print(":speech_balloon: [bold]请输入您的问题：[/] ", end="")
    query = console.input()
    logging.info(f"Original query:\n{query}")

    # 4) summarize & embed
    console.print(":hourglass: 正在提炼你的问题，并转化为文本嵌入")
    short_q = summarize_query(query, config)
    logging.info(f"Summarized query:\n{short_q}")

    q_emb = embed_text(short_q, config)
    logging.info(f"Embedded query computed")

    # 5) vector search
    res = search_top_k_with_scores(idx, q_emb, config.qa.top_k)
    topk_doc_int_ids, _ = zip(*res)
    topk_doc_hex_ids = [int2hex[int_id] for int_id in topk_doc_int_ids]
    topk_doc_paths = [processed[hex_id]['dst_file_path'] for hex_id in topk_doc_hex_ids]
    logging.info(f"{len(topk_doc_paths)} documents collected through retrieval "
                 f"via vector database:\n{'\n'.join(topk_doc_paths)}")
    console.print(f":pushpin: 根据文本嵌入，检索到最相关的{len(topk_doc_paths)}篇参考文档")

    # 6) augment by tags
    # build doc_id → tags map by loading intermediate .pkl
    extra_hex_ids = set()
    for i, doc_path in enumerate(topk_doc_paths):
        seed_hex_id = topk_doc_hex_ids[i]
        with open(doc_path, "rb") as fin:
            doc_data = pickle.load(fin)
            doc_tags = doc_data['tags']

        counter = Counter()
        for tag in doc_tags:
            for hex_id in inverted.get(tag, []):
                counter[hex_id] += 1

        delta = []
        for doc_hex_id, overlap in counter.items():
            if doc_hex_id != seed_hex_id and overlap >= config.qa.tag_threshold:
                extra_hex_ids.add(doc_hex_id)
                delta.append(processed[doc_hex_id]['dst_file_path'])
        if delta:
            # logging.info(f"Tags {doc_tags} found in {doc_path}, and there are another "
            #              f"{len(delta)} documents that share at least {config.qa.tag_threshold} "
            #              f"tags in common:\n{'\n\t'.join(delta)}")
            logging.info(f"Tags {doc_tags} found in {doc_path}, and there are another "
                         f"{len(delta)} documents that share at least {config.qa.tag_threshold}")
        else:
            logging.info(f"Tags {doc_tags} found in {doc_path}. No other document "
                         f"share at least {config.qa.tag_threshold} tags in common")

    other_doc_paths = [processed[hex_id]['dst_file_path'] for hex_id in list(extra_hex_ids)]
    logging.info(f"Another {len(other_doc_paths)} documents collected through "
                 f"retrieval via tag matching")
    console.print(f":pushpin: 根据关键词标签，检索到可能相关的额外{len(other_doc_paths)}篇参考文档")

    # 7) cap total references
    temp_sum = len(topk_doc_paths) + len(other_doc_paths)
    if temp_sum > config.qa.max_ref_docs:
        console.print(f":scales: 总参考文档数目超过阈值，"
                      f"随机丢弃{temp_sum - config.qa.max_ref_docs}篇额外文档")
        random.seed(config.qa.random_seed)
        other_doc_paths = random.sample(other_doc_paths,
                                        len(other_doc_paths) - (temp_sum - config.qa.max_ref_docs))
    ref_doc_paths = topk_doc_paths + other_doc_paths
    logging.info(f"Final reference documents sampled:\n{'\n'.join(ref_doc_paths)}")
    console.print(f":book: 最终参考文档数：{len(ref_doc_paths)}")

    # 8) generate per-doc answers
    answers = []
    console.print(":robot_face: [bold]先开始针对每个参考文档生成回答[/]")
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for ans in tqdm(
            executor.map(
                generate_answer_for_doc, ref_doc_paths,
                [query] * len(ref_doc_paths), [config] * len(ref_doc_paths),
                [str(log_file)] * len(ref_doc_paths)
            ),
            total=len(ref_doc_paths),
            desc="Docs",
            leave=False
        ):
            answers.append(ans)

    # 9) hierarchical summarization
    console.print(":sparkles: [bold]整合所有回答[/]")
    final = recursive_summarize(query, answers, config)

    # 10) output
    console.rule("[bold green]最终回答[/]")
    console.print(final, "\n")

    # 11) done
    end_time = time.perf_counter()
    duration = round(end_time - begin_time, 4)
    console.print(f":white_check_mark: [green]已完成，"
                  f"用时{duration}秒，详细日志见 {log_file}[/]")
    logging.info(f"Done in {duration}s.")


if __name__ == "__main__":
    main()
