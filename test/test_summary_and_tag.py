import dotenv
from munch import DefaultMunch
from construct import process_file


config_dict = {
    "paths": {
        "vector_index": "./knowledge/faiss_index.bin",
        "processed_track": "./knowledge/processed_files.json",
        "raw_markdown": "./knowledge/raw",
        "intermediate": "./knowledge/intermediate"
    },
    "models": {
        "summarization": "ep-20250212105505-5zlbx",  # doubao 1.5 pro
        "text_embedding": "doubao-embedding-text-240715",
        "tag_extraction": "ep-20250212105505-5zlbx"
    },
    "generative": {
        "max_summary_tokens": 4096
    }
}
config = DefaultMunch.fromDict(config_dict)


# src_file_path = "knowledge/raw/专业技术/(20191105)是否应当鼓励计算机专业的本科生积极参与科研_Yan Gu.md"  # 该内容会被过滤
src_file_path = "../knowledge/raw/专业技术/(20170703)读博士有多苦_高飞.md"


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path='../.env', override=True)

    result, b = process_file(src_file_path, config)
