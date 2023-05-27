import re
import json
import random
from typing import Text, List
import pickle
import numpy as np
from collections import defaultdict


def read_json(file_path: Text):
    with open(file_path, "r") as file:
        json_data = file.read()
    return json.loads(json_data)


# START: Loading collected chains...
def read_metaqa_kb_for_traversal(kb_file):
    e1_r_map = defaultdict(list)
    with open(kb_file) as fin:
        for line in fin:
            line = line.strip()
            e1, r, e2 = line.split("|")
            e1_r_map[(e1, r)].append(e2)
            e1_r_map[(e2, r + "_inv")].append(e1)
    return e1_r_map


def gather_paths(all_subgraphs):
    train_chains = {}
    for ctr, (qid, all_chains) in enumerate(all_subgraphs.items()):
        all_inference_chains = set()
        for chains in all_chains:
            chains_st_en = chains['chains']
            for path_st_en in chains_st_en:
                if path_st_en == ('type.object.type', 'type.type.instance') or path_st_en == (
                        '22-rdf-syntax-ns#type', 'type.type.instance'):
                    continue
                if type(path_st_en) == str:
                    path_st_en = (path_st_en,)
                all_inference_chains.add(path_st_en)
        train_chains[qid] = all_inference_chains
    return train_chains


def get_inference_chains_from_KNN(qid2knns, train_chains, k=5):
    no_chain_counter = 0
    no_chain_qids = []
    num_chains = []
    qid2chains = {}
    for q_ctr, (qid, knns) in enumerate(qid2knns.items()):
        all_chains = []
        for knn in knns[:k]:
            if knn == qid:
                continue
            chains = train_chains[knn] if knn in train_chains else []
            all_chains += chains
        if len(all_chains) == 0:
            no_chain_counter += 1
            no_chain_qids.append(qid)
        #             print(test_qid2qstr[qid])
        all_chains_set = set()
        for chain in all_chains:
            all_chains_set.add(chain)
        qid2chains[qid] = all_chains_set
        num_chains.append(len(all_chains_set))
    print("#Queries with no chains: {} out of {} questions, {:.2f}%".format(no_chain_counter, len(qid2knns),
                                                                            100 * (no_chain_counter / len(qid2knns))))
    print("Avg number of chains: {}".format(np.mean(num_chains)))
    return qid2chains, no_chain_qids


def get_triples_from_path(path: List[str]):
    prev_ent = None
    prev_rel = None
    triples = set()
    for ctr, e_or_r in enumerate(path):
        if ctr % 2 == 0:  # entiy
            if prev_ent is not None and prev_rel is not None:
                triples.add((prev_ent, prev_rel, e_or_r))
            prev_ent = e_or_r
        else:
            prev_rel = e_or_r
    return triples


def get_subgraph_spanned_by_chain(e: str, path: List[str], depth: int, max_branch: int, dataset_name=None,
                                  e1_r_map=None):
    """
    starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
    max_branch number. We also get the intermediate entities and relations
    """
    if depth == len(path):
        # reached end, return node
        return [[e]]
    next_rel = path[depth]
    #     next_entities = subgraph[(e, path[depth])]
    if dataset_name.lower() == "metaqa":
        next_entities = e1_r_map[(e, next_rel)]
    # next_entities = list(set(self.train_map[(e, path[depth])] + self.args.rotate_edges[(e, path[depth])][:5]))
    if len(next_entities) == 0:
        # edge not present
        return []
    if len(next_entities) > max_branch:
        # select max_branch random entities
        next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
    suffix_chains = []
    for e_next in next_entities:
        paths_from_branch = get_subgraph_spanned_by_chain(e_next, path, depth + 1, max_branch, dataset_name, e1_r_map)
        for p in paths_from_branch:
            suffix_chains.append(p)
    # now for each chain, append (the current entity, relations)
    temp = []
    for chain in suffix_chains:
        if len(chain) == 0:  # these are the chains which didnt execute, ignore them
            continue
        temp.append([e, path[depth]] + chain)
    suffix_chains = temp
    return suffix_chains


def collect_subgraph_execute_chains(chains, qid2qents, job_id, total_jobs, dataset_name=None, e1_r_map=None):
    depth = 0
    max_branch = 100
    subgraph_lengths = []
    triples_all_qs = {}
    job_size = len(chains) / total_jobs
    st = job_id * job_size
    en = (1 + job_id) * job_size
    print("St: {}, En: {}".format(st, en))
    # sort chains wrt qids so that every partition gets the same list
    chains = [(qid, q_chains) for (qid, q_chains) in sorted(chains.items(), key=lambda item: item[0])]
    if dataset_name == "metaqa":
        assert e1_r_map is not None
    for ctr, (qid, q_chains) in enumerate(chains):
        if st <= ctr < en:
            q_ents = qid2qents[qid]
            all_triples = set()
            for q_ent in q_ents:
                for chain in q_chains:
                    all_executed_chains = get_subgraph_spanned_by_chain(q_ent, chain, depth, max_branch, dataset_name,
                                                                        e1_r_map)
                    for executed_chain in all_executed_chains:
                        triples = get_triples_from_path(executed_chain)
                        all_triples = all_triples | triples
            subgraph_lengths.append(len(all_triples))
            triples_all_qs[qid] = all_triples
    return triples_all_qs, subgraph_lengths


def load_adaptive_graph(item_test, k):
    # Loading collected chains...
    collected_chains_file_path = "../adaptive_subgraph_collection/data/subgraph/metaqa_train_chains_0.pkl"
    kb_file_path = "/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/kb.txt"

    with open(collected_chains_file_path, "rb") as fin:
        all_subgraphs = pickle.load(fin)
    train_chains = gather_paths(all_subgraphs)

    qid2qents = {item_test["id"]: item_test["seed_entities"]}
    qid2knns = {item_test["id"]: item_test["knn"]}
    qid2chains, _ = get_inference_chains_from_KNN(qid2knns, train_chains, k)

    # Executing collected chains for subgraph...
    e1_r_map = read_metaqa_kb_for_traversal(kb_file_path)
    triples_all_qs, subgraph_lengths = collect_subgraph_execute_chains(qid2chains, qid2qents, 0, 1, 'metaqa', e1_r_map)
    return triples_all_qs


# END: Loading collected chains...


def process_answer(input_text, intent, random_knn=20):
    train_data_path = "/home/namnv/Documents/FTECH/Project/research/CBR-SUBG-KBQA/adaptive_subgraph_collection/data/train.json"
    train_data = read_json(train_data_path)
    train_data = [data["id"] for data in train_data if data["intent"] == intent]

    matches = re.findall(r"\[(.*?)\]", input_text)
    seed_entities = [match for match in matches]

    item = {
        "id": "test_0",
        "seed_entities": seed_entities,
        "question": input_text,
        "answer": ["1997-02-27"],
        "intent": intent,
        "knn": random.sample(train_data, random_knn)
    }

    load_adaptive_graph(item, k=random_knn)
    return item


import yaml
if __name__ == "__main__":
    # input_text = "năm bao nhiêu cô [Ngô Phượng Hoàng] sinh ra ợ"
    # intent = "faq/knowledge_ask_employee_birthday_fullname"
    #
    # process_answer(input_text, intent)
    # load_adaptive_graph()

    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
