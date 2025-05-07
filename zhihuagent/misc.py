import os
import re
import json
import yaml
import logging
from munch import DefaultMunch


class MyConfig(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def load_config(working_dir):
    config_path = os.path.join(working_dir, "config.yml")
    with open(config_path, 'rb') as fin:
        config_dict = yaml.load(fin, Loader=yaml.FullLoader)

    config_dict['working_dir'] = os.path.abspath(working_dir)
    result = DefaultMunch.fromDict(config_dict)
    return result


def set_log(filename=None):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # otherwise, run_gaia will produce messy logs

    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='[%(levelname)s][%(asctime)s.%(msecs)03d][%(process)d]'
               '[%(filename)s:%(lineno)d]: %(message)s',
        datefmt='(%Y-%m-%d) %H:%M:%S'
    )
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)


def sanitize_json(text):
    out, in_str, escape = [], False, False
    i, n = 0, len(text)
    # Sanitize JSON: escape interior double‑quotes and strip any backslash+'
    # so it becomes a literal single‑quote, preserving all other escapes.
    while i < n:
        ch = text[i]
        if not in_str:
            if ch == '"': in_str = True
            out.append(ch)

        else:
            if escape:
                out.append(ch)
                escape = False

            elif ch == '\\':
                # if next is a single‑quote, drop the backslash
                if i + 1 < n and text[i+1] == "'":
                    out.append("'")
                    i += 1
                else:
                    out.append(ch)
                    escape = True

            elif ch == '"':
                # lookahead for a true closer
                j = i + 1
                while j < n and text[j].isspace(): j += 1
                if j < n and text[j] in {':', ',', '}', ']'}:
                    out.append(ch)
                    in_str = False
                else:
                    out.append(r'\"')

            else:
                out.append(ch)

        i += 1

    return ''.join(out)


def extract_json_from_string(text):
    text = sanitize_json(text)
    # logging.info(f"\nSanitized text:\n{text}")

    # json_regex = r'```json\n\s*\{\n\s*[\s\S\n]*\}\n\s*```'
    json_regex = r'```json\n\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\n\s*```'
    matches = re.findall(json_regex, text)
    if matches:
        json_data = matches[0].replace('```json', '').replace('```', '').strip()
        try:
            # Parse the JSON data
            parsed_json = json.loads(json_data)
            return parsed_json
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON data: {e}")
    else:
        logging.error(f"No JSON data found")

    return {}
