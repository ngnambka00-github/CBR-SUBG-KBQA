import os
import json
import pickle
import argparse
from typing import *

from tqdm import tqdm
from numpy.random import default_rng
import numpy as np

from adaptive_subgraph_collection.adaptive_utils import get_query_entities_and_answers_metaqa, execute_kb_query, \
    read_metaqa_kb_for_traversal, read_yaml

rng = default_rng()


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
    else:
        spql = "select distinct ?e where { ns:" + e + " ns:" + next_rel + " ?e .}"
        ret = execute_kb_query(spql)
        next_entities = ret[0]
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
    for ctr, (qid, q_chains) in tqdm(enumerate(chains), total=len(chains)):
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


def load_knns(file_name):
    #     test_file_name = os.path.join(dir_name, "test_roberta-base_mean_pool_masked.json")
    with open(file_name) as fin:
        data = json.load(fin)
    qid2knns, qid2qstr = {}, {}
    for t in data:
        qid = t["id"]
        knn_qs = t["knn"]
        qid2knns[qid] = knn_qs
    return qid2knns


def get_inference_chains_from_KNN(qid2knns, train_chains, k=5):
    no_chain_counter = 0
    no_chain_qids = []
    num_chains = []
    qid2chains = {}
    for q_ctr, (qid, knns) in enumerate(tqdm(qid2knns.items())):
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


def check_overlap_inference_chains(qid2chains, qid2gold_chains):
    gold_chain_present_ctr = 0
    for qid, gold_chains in qid2gold_chains.items():
        for gold_chain in gold_chains:
            if qid in qid2chains and gold_chain in qid2chains[qid]:
                gold_chain_present_ctr += 1
                break
    print("Gold inference chain present for {} out of {} questions, {:.2f}%".format(gold_chain_present_ctr,
                                                                                    len(qid2chains), 100 * (
                                                                                            gold_chain_present_ctr / len(
                                                                                        qid2chains))))


def check_overlap(triples_all_qs, qid2answers):
    total_ctr, ans_present_ctr = 0, 0
    wrong_qid_answers = []
    for ctr, (qid, triples) in tqdm(enumerate(triples_all_qs.items())):
        flag = 0
        q_entities = set()
        for (e1, r, e2) in triples:
            q_entities.add(e1)
            q_entities.add(e2)
        for ans_e in qid2answers[qid]:
            if ans_e in q_entities:
                flag = 1
                ans_present_ctr += 1
                break
        total_ctr += 1
        if flag == 0:
            wrong_qid_answers.append(qid)
    print("Answer present for {} questions out of {} questions, {:.2f}%".format(ans_present_ctr, total_ctr,
                                                                                100 * (ans_present_ctr / total_ctr)))
    return wrong_qid_answers


# Example run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities using CBR")
    parser.add_argument("--config", type=str, default='../config/config.yaml')
    args = parser.parse_args()
    config = read_yaml(args.config)["DATA_COLLECTION"]["adaptive_graph_collection"]

    print("Loading collected chains...")
    with open(config["collected_chains_file"], "rb") as fin:
        all_subgraphs = pickle.load(fin)
    train_chains = gather_paths(all_subgraphs)

    print("Getting query entities and answers....")
    if config["dataset_name"] == 'metaqa':
        qid2qents, qid2answers, _, _ = get_query_entities_and_answers_metaqa(config["input_file"])

    print("Loading KNNs...")
    qid2knns = load_knns(config["input_file"])

    print("Getting inference chains...")
    qid2chains, no_chain_qids = get_inference_chains_from_KNN(qid2knns, train_chains, k=int(config["k"]))

    print("Executing collected chains for subgraph...")
    e1_r_map = None
    if config["dataset_name"] == "metaqa":
        e1_r_map = read_metaqa_kb_for_traversal(config["kb_file"])
    triples_all_qs, subgraph_lengths = collect_subgraph_execute_chains(qid2chains, qid2qents,
                                                                       int(config["job_id"]), int(config["total_jobs"]),
                                                                       config["dataset_name"], e1_r_map)
    print("Saving")
    out_file_name = f"{config['dataset_name']}_cbr_subgraph_{config['split']}_{config['k']}.pkl"
    with open(os.path.join(config['out_dir'], out_file_name), "wb") as f_out:
        pickle.dump(triples_all_qs, f_out)

    # print(f"CHECK triples_all_qs: {triples_all_qs}")
    print(f"File written to {os.path.join(config['out_dir'], out_file_name)}")
    if len(qid2answers) > 0:  # e.g. CWQ test set has no answers
        wrong_qid_answers = check_overlap(triples_all_qs, qid2answers)
