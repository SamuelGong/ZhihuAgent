import os
import re
import glob
import json
import time
import math
import faiss
import base64
import dotenv
import pickle
import hashlib
import logging
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from zhihuagent.model_related import get_model_response, get_embedding
from zhihuagent.misc import load_config, set_log, extract_json_from_string


def load_processed(processed_track_path):
    if not processed_track_path.exists():
        return set()
    return set([
        e["src_file_path"] for e
        in json.loads(processed_track_path.read_text()).values()
    ])


def list_markdown_files(raw_dir):
    result = []
    for p in glob.glob(str(raw_dir / "**" / "*.md"), recursive=True):
        result.append(Path(p))
    return result


# def sanitize_text(text):
#     # 1) Remove any Markdown links [text](url), including broken ones like [[url]([url)
#     text = re.sub(r'\[([^\]]*?)\]\([^\)]*?\)', ' ', text)
#     # 2) (Optional) Remove any remaining bare URLs
#     text = re.sub(r'https?://\S+', ' ', text)
#     # 3) Collapse runs of whitespace into a single space
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     return text


def summarize_text(text, config):
    # text = sanitize_text(text)
    messages = [
        {"role": "system", "content": "你善于总结 Markdown 内的文本内容。"},
        {"role": "user", "content": f"请用三到五句话总结以下内容:\n\n{text}"}
    ]
    # print(f"DEBUG: {len(text)} {messages}")
    result, finish_reason = get_model_response(
        model=config.models.summarization,
        messages=messages,
        temperature=0.0,
        max_tokens=config.generative.max_summary_tokens
    )
    return result, finish_reason


def describe_image(image_path, config):
    image_path = Path(image_path)
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        content_type = 'image/jpeg'
    elif ext in ['.apng', '.png']:
        content_type = 'image/png'
    elif ext in ['.gif']:
        content_type = 'image/gif'
    elif ext in ['.webp']:
        content_type = 'image/webp'
    elif ext in ['.bmp', '.dib']:
        content_type = 'image/bmp'
    elif ext in ['.tiff', '.tif']:
        content_type = 'image/tiff'
    elif ext in ['.ico']:
        content_type = "image/x-icon"
    elif ext in ['.icns']:
        content_type = "image/icns"
    elif ext in ['.sgi']:
        content_type = "image/sgi"
    elif ext in ['.j2c', '.j2k', '.jp2', '.jpc', '.jpf', '.jpx']:
        content_type = "image/jp2"
    else:
        raise NotImplementedError
    image_url = f"data:{content_type};base64,{img_base64}"

    messages = [
        {"role": "system", "content": "你善于对图片生成准确清晰的描述。"},
        {"role": "user", "content": [
            {"type": "text", "text": "描述这张图片。"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}
    ]
    result, _ = get_model_response(
        model=config.models.vision_caption,
        messages=messages
    )
    return result


def extract_tags(summary, config):
    messages = [
        {"role": "system", "content": "你善于从文本中提取通用的、便于用以搜索语义相近文章的关键词标签。"},
        {"role": "user", "content": f"请为以下内容提取五到十个关键词标签:\n\n{summary}"
                                    f"\n\n请只生成最终的标签列表：标签1, 标签2, ..."}
    ]
    result, _ = get_model_response(
        model=config.models.tag_extraction,
        messages=messages,
        temperature=0.5
    )

    if result.startswith('['):
        result = result[1:]
    if result.endswith(']'):
        result = result[:-1]
    result = result.replace('，', ',')
    tags = result.split(",")
    return [t.strip().lower() for t in tags if t.strip()]


def embed_text(text, config):
    result = get_embedding(
        text=text,
        model=config.models.text_embedding
    )
    return result


def process_file(src_file_path, config):
    try:
        set_log()

        src_file_path = Path(src_file_path)

        raw_dir = Path(config.paths.raw_markdown)
        rel = src_file_path.relative_to(raw_dir)
        intermediate_dir = Path(config.paths.intermediate)
        dst_file_path = (intermediate_dir / rel).with_suffix(".pkl")

        if dst_file_path.exists():
            logging.info(f"Skipping {src_file_path}: "
                         f"intermediate file exists at {dst_file_path}")
            with open(dst_file_path, "rb") as f:
                result = pickle.load(f)
            return result, False

        """Read, summarize, tag, embed a single markdown file."""
        text = src_file_path.read_text(encoding="utf-8")

        # find local images: ![alt](relative/path.png)
        img_paths = []
        for m in re.finditer(r"!\[.*?\]\((.*?)\)", text):
            img = (src_file_path.parent / m.group(1)).resolve()
            if img.exists():
                img_paths.append(img)

        txt_summary, finish_reason = summarize_text(text, config)
        if not finish_reason == "stop":
            logging.info(f"Skipping {src_file_path}: text summarization "
                         f"aborted due to {finish_reason}")
            result = {}
            return result, False

        img_summaries = [describe_image(p, config) for p in img_paths]
        full_summary = "\n\n".join([txt_summary] + img_summaries)
        tags = extract_tags(full_summary, config)
        embedding = embed_text(full_summary, config)

        # ID = hash of relative path
        raw_dir = Path(config.paths.raw_markdown)
        doc_id = hashlib.sha1(str(src_file_path.relative_to(raw_dir)).encode()).hexdigest()
        result = {
            "id": doc_id,
            "src_file_path": str(src_file_path),
            "dst_file_path": str(dst_file_path),
            "original": text,
            "summary": full_summary,
            "tags": tags,
            "embedding": embedding
        }

        out_dir = dst_file_path.parent
        if not out_dir.exists():
            os.makedirs(out_dir)
        with open(dst_file_path, 'wb') as fout:
            pickle.dump(result, fout)
    except Exception as e:
        logging.info(f"Error processing {src_file_path} due to {e}/{traceback.format_exc()}")
    return result, True


def merge_similar_tags(inverted, config):
    """
    Ask the LLM to cluster semantically similar tags, then merge their doc-ID lists.
    Returns a new inverted index mapping each canonical tag to the unioned doc IDs.
    """
    system_prompt = """你是一个标签分群助手，负责将语义相近的标签聚合。
对于下列一组关键词标签，请将语义相近的标签合并为同一组，并输出一个 JSON 映射：
key 为该组的“规范化标签”（挑选一个最通用的词），value 为该组所有原始标签列表。

请严格遵从该输出格式：
```json
{
    "规范化标签1": ["原始标签1", "原始标签2", ...],
    "规范化标签2": ["原始标签3", "原始标签4", ...]
}
```

注意：
1. 该 JSON 映射不必包含所有原始标签。对于原本已经不和任何其他标签语义相近的标签，默认保持原样。
2. 规范化标签不必包含“相关”两个字，这两个字是冗余的
3. 因为原始标签过多，尽量多做合并以显著减少规范化标签的数量
"""

    user_query = f"关键词标签：{list(inverted.keys())}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query}
    ]

    while True:
        resp, _ = get_model_response(
            model=config.models.tag_extraction,
            messages=messages,
            temperature=0.0
        )
        # print(resp)

        new2old_tag_map = extract_json_from_string(resp)
        if not new2old_tag_map:
            logging.error("Tag-merge LLM 返回内容无法解析为 JSON，重试合并。内容：\n%s", resp)
            continue
        break

    new_inverted = {}
    seen_in_clusters = set()

    # For each cluster, union its postings
    for new_tag, old_tags in new2old_tag_map.items():
        merged_doc_ids = set()
        for old_tag in old_tags:
            merged_doc_ids.update(inverted.get(old_tag, []))
            seen_in_clusters.add(old_tag)
        new_inverted[new_tag] = sorted(merged_doc_ids)

    # Any tags the LLM didn’t mention get carried over as-is
    for t, doc_ids in inverted.items():
        if t not in seen_in_clusters:
            new_inverted[t] = doc_ids

    return new_inverted, new2old_tag_map


def _chunk_iterable(it, size):
    """Yield successive chunks (as lists) of length ≤ size."""
    it = iter(it)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def _merge_chunk(sub_inv, config):
    _, cmap = merge_similar_tags(sub_inv, config)
    return cmap


def multi_level_merge_similar_tags(
    inverted,
    config,
    chunk_size: int = 300,
    final_threshold: int = 400
):
    """
    Repeatedly chunk‐merge until len(tags) <= final_threshold,
    then do one final LLM pass. Returns (final_inv, new2old_map).
    """
    # 1) identity map
    overall_map = { t: [t] for t in inverted }
    current_inv = inverted.copy()
    level = 1

    while True:
        tags = list(current_inv.keys())
        n_tags = len(tags)
        # stop condition
        if n_tags <= final_threshold:
            logging.info(f"[TagMerge] Level {level}: {n_tags} tags ≤ final_threshold ({final_threshold}); stopping recursion.")
            break

        n_chunks = math.ceil(n_tags / chunk_size)
        logging.info(f"[TagMerge] Level {level}: {n_tags} tags → {n_chunks} chunks (≤{chunk_size} each)")

        # prepare sub‐inverted dicts
        sub_invs = [
            { t: current_inv[t] for t in tags[i*chunk_size:(i+1)*chunk_size] }
            for i in range(n_chunks)
        ]

        # parallel merges
        partial_maps = []
        with ProcessPoolExecutor(max_workers=cpu_count()) as exec:
            for cmap in tqdm(
                exec.map(_merge_chunk, sub_invs, [config]*n_chunks),
                total=n_chunks,
                desc=f"TagMerge L{level}",
                leave=False
            ):
                if cmap:
                    partial_maps.append(cmap)

        # flatten & dedupe partial_maps → next_level_map
        next_level_map = {}
        for cmap in partial_maps:
            for canon, members in cmap.items():
                next_level_map.setdefault(canon, []).extend(members)
        for canon, members in next_level_map.items():
            next_level_map[canon] = list(dict.fromkeys(members))

        # build next_inv (canonical → merged doc‐ids)
        next_inv = {}
        for canon, olds in next_level_map.items():
            docs = set()
            for t in olds:
                docs.update(current_inv.get(t, []))
            next_inv[canon] = sorted(docs)

        # compose overall_map: new canon → all original tags
        new_overall = {}
        for canon, olds in next_level_map.items():
            flat = []
            for t in olds:
                flat.extend(overall_map.get(t, [t]))
            new_overall[canon] = list(dict.fromkeys(flat))

        # advance
        overall_map = new_overall
        current_inv = next_inv
        level += 1

    # final single‐pass merge on the small set
    if len(current_inv) > 1:
        logging.info(f"[TagMerge] Final pass on {len(current_inv)} tags")
        current_inv, final_cmap = merge_similar_tags(current_inv, config)

        # expand overall_map via final_cmap
        final_map = {}
        for canon, members in final_cmap.items():
            flat = []
            for m in members:
                flat.extend(overall_map.get(m, [m]))
            final_map[canon] = list(dict.fromkeys(flat))
        overall_map = final_map

    return current_inv, overall_map


def update_intermediate_tags(new2old_tag_map, intermediate_root):
    old2new_tag_map = {}
    for new_tag, old_tags in new2old_tag_map.items():
        for old_tag in old_tags:
            old2new_tag_map[old_tag] = new_tag

    int_root = Path(intermediate_root)
    for pkl_path in tqdm(int_root.rglob("*.pkl"), desc="Updating intermediate tags"):
        with open(pkl_path, "rb") as f:
            doc = pickle.load(f)

        orig_tags = doc.get("tags", [])
        new_tags = list(
            dict.fromkeys(
                old2new_tag_map.get(t, t)
                for t in orig_tags
            ))

        doc["tags"] = new_tags
        with open(pkl_path, "wb") as f:
            pickle.dump(doc, f)


def main():
    begin_time = time.perf_counter()

    dotenv.load_dotenv(dotenv_path='.env', override=True)
    config = load_config(os.getcwd())
    set_log()
    raw_markdown_path = Path(config.paths.raw_markdown)
    processed_track_path = Path(config.paths.processed_track)

    logging.info("Scanning for markdown files")
    seen = load_processed(processed_track_path)
    all_md = list_markdown_files(raw_markdown_path)
    to_process = [p for p in all_md if str(p) not in seen]
    # to_process = all_md  # DEBUG
    if not to_process:
        logging.info("No new files to process. Exiting.")
        return
    else:
        logging.info(f"Processing {len(to_process)} files.")
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(
            tqdm(executor.map(
                process_file,
                to_process,
                [config] * len(to_process)),
                total=len(to_process)
            )
        )
        # remove empty result, probably due to content filter
        results = [(res, is_new) for res, is_new in results if len(res) > 0]

    # ── Load or create inverted index ────────────────────────────────────────
    inverted_index_path = Path(config.paths.inverted_index)
    if inverted_index_path.exists():
        inverted = json.loads(inverted_index_path.read_text())
        logging.info("Loaded existing inverted index.")
    else:
        inverted = {}
        logging.info("Creating new inverted index.")

    # ── Load or create FAISS index ─────────────────────────────────────────
    vector_index_path = Path(config.paths.vector_index)
    if vector_index_path.exists():
        idx = faiss.read_index(str(vector_index_path))
        logging.info("Loaded existing FAISS index.")
    else:
        embedding_dim = len(results[0][0]["embedding"])
        hnsw = faiss.IndexHNSWFlat(embedding_dim, config.vector_db.m)
        hnsw.hnsw.efConstruction = config.vector_db.c
        hnsw.hnsw.efSearch = config.vector_db.s
        idx = faiss.IndexIDMap(hnsw)
        logging.info("Creating new FAISS index.")

    # ── Traversal and update ───────────────────────────────────────────────
    new_embedding_list = []
    new_id_list = []
    id2path_map = {}  # for post-processing after vector database querying
    for result, is_new in results:
        id2path_map.update({result["id"]: {
            "src_file_path": result["src_file_path"],
            "dst_file_path": result["dst_file_path"]
        }})
        if is_new:
        # if True:  # DEBUG
            new_embedding_list.append(result["embedding"])
            new_id_list.append(int(int(result["id"], 16) % (10**12)))  # FAISS likes this
            for tag in result["tags"]:
                inverted.setdefault(tag, []).append(result["id"])
    new_embedding_list = np.array(new_embedding_list)
    new_id_list = np.array(new_id_list)

    processed_track_path.write_text(json.dumps(id2path_map, indent=4, ensure_ascii=False))
    logging.info(f"The mapping of document IDs to paths written "
                 f"to {str(config.paths.processed_track)}.")

    idx.add_with_ids(new_embedding_list, new_id_list)
    faiss.write_index(idx, str(vector_index_path))
    logging.info(f"The vector database written to {str(vector_index_path)}.")

    # ── Update tags, and thus inverted index, if necessary  ────────────────
    logging.info("Optimizing tags via LLM")
    inverted, new2old_tag_map = multi_level_merge_similar_tags(
        inverted,
        config,
        chunk_size=config.tag_merge.chunk_size,  # adjust to your LLM’s safe prompt-size
        final_threshold=config.tag_merge.final_threshold # or more if you have huge tag sets
    )
    inverted_index_path.write_text(json.dumps(inverted, indent=4, ensure_ascii=False))
    logging.info(f"Final inverted index written to {str(inverted_index_path)}.")

    if new2old_tag_map:
        logging.info("Updating all intermediate .pkl files with optimized tags")
        update_intermediate_tags(new2old_tag_map, config.paths.intermediate)

    end_time = time.perf_counter()
    duration = round(end_time - begin_time, 4)
    logging.info(f"Updated processed files record.\nDone in {duration}s")


if __name__ == "__main__":
    main()
