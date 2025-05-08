#!/usr/bin/env python3
import os
import sys
import time
import dotenv
import logging
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zhihuagent.model_related import get_model_response


def main():
    begin_time = time.perf_counter()

    dotenv.load_dotenv(dotenv_path='../.env', override=True)
    console = Console()

    console.print(":mag_right: [bold yellow]我是只使用大语言模型回答问题的助手[/]")
    console.print(":speech_balloon: [bold]请输入您的问题：[/] ", end="")
    query = console.input()

    console.print(":hourglass: 正在使用 DeepSeek-R1，而不基于任何参考文档生成回答")

    system_prompt = """你是一个有用的问答助手。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"原始问题：{query}\n\n"
            "请回答原始问题："
         }
    ]
    resp, _ = get_model_response(
        model="ep-20250210212226-nffpf",
        messages=messages,
        temperature=1.0,
        # max_tokens=config.generative.max_answer_tokens
    )
    final = resp.strip()

    # 10) output
    console.rule("[bold green]最终回答[/]")
    console.print(final, "\n")

    # 11) done
    end_time = time.perf_counter()
    duration = round(end_time - begin_time, 4)
    console.print(f":white_check_mark: [green]已完成，"
                  f"用时{duration}秒[/]")
    logging.info(f"Done in {duration}s.")


if __name__ == "__main__":
    main()
