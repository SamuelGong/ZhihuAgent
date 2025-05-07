import dotenv
from munch import DefaultMunch
from main import generate_answer_for_doc

doc_path = "../knowledge/intermediate/专业技术/(20240126)怎样才能确定女朋友是否适合结婚_养心·随缘而行.pkl"
query = "找老婆的标准是什么？"

config_dict = {
    "models": {
        "dialog": "ep-20250212105505-5zlbx"  # doubao 1.5 pro
    }
}
config = DefaultMunch.fromDict(config_dict)


if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path='../.env', override=True)
    answer = generate_answer_for_doc(
        doc_path=doc_path,
        query=query,
        config=config,
        log_file=None
    )
    print(answer)
