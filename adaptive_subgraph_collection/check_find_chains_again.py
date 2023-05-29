import os
import pickle
import argparse
from collections import defaultdict

from tqdm import tqdm
from numpy.random import default_rng

from adaptive_subgraph_collection.adaptive_utils import get_query_entities_and_answers_metaqa, \
    read_metaqa_kb, find_paths, read_yaml

rng = default_rng()


def check_has_path(q_entity, a_entity):
    e1_map = read_metaqa_kb(
        "/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/kb.txt")
    return find_paths(e1_map, q_entity, a_entity)


def refactor_all_chains(
        old_path="/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/subgraph_2/metaqa_train_chains_0.pkl",
        new_path=None
):
    e1_map = read_metaqa_kb(
        "/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/kb.txt")
    with open(old_path, "rb") as f_in:
        all_subgraphs = pickle.load(f_in)

    for key, chains in all_subgraphs.items():
        for chain in chains:
            if len(chain["chains"]) == 0:
                print(f"Before: {chains}")
                q_ent = chain["st"]
                ans_ent = chain["en"]
                paths = find_paths(e1_map, q_ent, ans_ent)
                if len(paths) == 0:
                    for i in range(5):
                        paths = find_paths(e1_map, q_ent, ans_ent)
                        if len(paths) != 0:
                            break
                chain["chains"] = paths
                print(f"After: {chains}\n")

    if new_path:
        with open(new_path, "wb") as f_out:
            pickle.dump(all_subgraphs, f_out)


if __name__ == '__main__':
    # q_ent = "Hà Nội"
    # ans_ent = "Tầng 5, 8 và 9, tòa nhà HTP Building, số 434 Trần Khát Chân, Hai Bà Trưng, Hà Nội"
    # print(check_has_path(q_ent, ans_ent))

    refactor_all_chains(
        # old_path="/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/subgraph/metaqa_train_chains_0.pkl",
        old_path="/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/subgraph/metaqa_train_chains_1.pkl",
    )
