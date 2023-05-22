import os
import re
from typing import Text, Dict, List, Union

from file_utils import read_yaml_to_dict


def read_all_alias(folder_path: Text) -> Dict[Text, List[Text]]:
    assert os.path.isdir(folder_path), "Folder path is not existed"
    alias_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(folder_path, file_name)
            alias_in_file = read_yaml_to_dict(file_path)
            alias_dict.update(alias_in_file)
    return alias_dict


# anh @[shortname_fullname] có tên đầy đủ là gì|@[1]
# cho hỏi @[role_department_fullname] @[1] có tên đầy đủ là gì|@[2]
def fill_alias_to_text(text: Text, alias: Dict[Text, List[Text]]) -> Union[List[Text], None]:
    match = re.search(r"\@\[(.*?)\]", text)
    if match:
        alias_key = match.group(1)
        assert alias_key in alias.keys(), "Alias is not valid"

        data = []
        alias_values = alias[alias_key]
        for a in alias_values:
            als = a.split("|")
            new_text = re.sub(f"\@\[{alias_key}\]", als[0], text)
            for i in range(1, len(als)):
                new_text = re.sub(f"\@\[{i}\]", als[i], new_text)
            data.append(new_text)
        return data
    return None


if __name__ == "__main__":
    alias_folder_path = "./data/alias/"
    alias = read_all_alias(folder_path=alias_folder_path)
    # input_text = "anh @[shortname_fullname] có tên đầy đủ là gì|@[1]"
    input_text = "@[role_department_fullname] @[1] có tên đầy đủ là gì|@[2]"
    print(fill_alias_to_text(input_text, alias))
