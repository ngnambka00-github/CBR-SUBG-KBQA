from typing import Text, List, Dict
import yaml

import pandas as pd


def read_csv(file_path: Text) -> pd.DataFrame:
    return pd.read_csv(file_path)


def write_to_txt(data: List[Text], out_path: Text) -> None:
    with open(out_path, "w") as fw:
        for data_item in data:
            fw.write(f"{data_item}\n")
    print(f"Write data to {out_path} successfully!!")


def read_from_txt(file_path: Text) -> List[Text]:
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [d.strip() for d in data]


def write_dict_to_yaml(data: Dict[Text, List[Text]], out_path: Text) -> None:
    with open(out_path, mode="w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
    print(f"Write data to {out_path} successfully!!")
