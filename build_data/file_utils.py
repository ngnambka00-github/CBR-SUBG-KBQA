from typing import Text, List, Dict
import yaml
import re

import pandas as pd


def read_csv(file_path: Text) -> pd.DataFrame:
    return pd.read_csv(file_path)


def write_to_txt(data: List[Text], out_path: Text) -> None:
    with open(out_path, "w") as fw:
        for data_item in data:
            fw.write(f"{data_item}\n")
    print(f"Write data to {out_path} successfully!!")


def write_dict_to_txt(data: Dict[Text, List], out_path: Text) -> None:
    with open(out_path, "w") as fw:
        for intent, examples in data.items():
            fw.write(f"~[{intent}]\n")
            for example in examples:
                _temp = re.sub(r'\|', r'\|', example)
                _temp = re.sub(r'\/', r'\/', _temp)
                _temp = re.sub(r'\@', r'\@', _temp)

                pattern = r"*\{" + _temp + r"\}"
                fw.write(f"\t{pattern}\n")
            fw.write("\n")
    print(f"Write data to {out_path} successfully!!")


def read_from_txt(file_path: Text) -> List[Text]:
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [d.strip() for d in data]


def write_dict_to_yaml(data: Dict[Text, List[Text]], out_path: Text) -> None:
    with open(out_path, mode="w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
    print(f"Write data to {out_path} successfully!!")


def read_yaml_to_dict(in_path: Text) -> Dict[Text, List[Text]]:
    with open(in_path, "r", encoding="utf-8") as file_reader:
        data = yaml.load(file_reader, Loader=yaml.FullLoader)
    return data
