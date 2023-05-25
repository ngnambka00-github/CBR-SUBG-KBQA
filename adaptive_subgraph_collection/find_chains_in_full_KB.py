import os
from collections import defaultdict
from tqdm import tqdm
import pickle
from numpy.random import default_rng
import numpy as np
import argparse
import wandb

rng = default_rng()
from adaptive_subgraph_collection.adaptive_utils import get_query_entities_and_answers, \
    get_query_entities_and_answers_cwq, execute_kb_query_for_hops, get_query_entities_and_answers_freebaseqa, \
    get_query_entities_and_answers_metaqa, read_metaqa_kb, find_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities using CBR")
    parser.add_argument("--train_file", type=str,
                        default='/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/data_with_mentions/webqsp_data_with_mentions/train.json')
    parser.add_argument("--dataset_name", type=str, default='webqsp')
    parser.add_argument("--output_dir", type=str,
                        default='/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/subgraphs/webqsp_gold_entities')
    parser.add_argument("--use_gold_entities", action='store_true')
    parser.add_argument("--metaqa_kb_file", type=str, default="/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/MetaQA-synthetic/3-hop/kb.txt")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=1)
    args = parser.parse_args()
    # args.use_wandb = (args.use_wandb == 1)
    if args.use_wandb:
        wandb.init("adaptive-subgraph-collection")

    if args.dataset_name.lower() == 'metaqa':
        qid2qents, qid2answers, qid2gold_spqls, qid2q_str = get_query_entities_and_answers_metaqa(args.train_file,
                                                                                                  return_gold_entities=args.use_gold_entities)
    # read entity
    # with open("./data/entity.txt", 'r') as file:
    #     entities_data = file.readlines()
    # entity2idx = {}
    # count = 0
    # for e in entities_data:
    #     e = e.strip()
    #     entity2idx[e] = count
    #     count += 1
    # idx2entity = {}
    # for k, v in entity2idx.items():
    #     idx2entity[v] = k

    if args.dataset_name.lower() == 'metaqa':  # metaqa has its own KB and not full Freebase, hence do not need SPARQL
        # read metaqa KB
        # find 1, 2, 3 hop paths between question entities and answers
        all_subgraphs = defaultdict(list)
        e1_map = read_metaqa_kb(args.metaqa_kb_file)
        qid2qents = [(qid, q_ents) for (qid, q_ents) in sorted(qid2qents.items(), key=lambda item: item[0])]
        job_size = len(qid2qents) / args.total_jobs
        st = args.job_id * job_size
        en = (1 + args.job_id) * job_size
        print("St: {}, En: {}".format(st, en))
        empty_ctr = 0
        all_len = []
        for ctr, (qid, q_ents) in tqdm(enumerate(qid2qents)):
            if st <= ctr < en:
                ans_ents = qid2answers[qid]
                len_q = 0
                for q_ent in q_ents:
                    for ans_ent in ans_ents:
                        paths = find_paths(e1_map, q_ent, ans_ent)
                        all_subgraphs[qid].append({'st': q_ent, 'en': ans_ent, 'chains': paths})
                        len_q += len(paths)
                if len_q == 0:
                    empty_ctr += 1
                all_len.append(len_q)

        print(f"CHECK: {all_subgraphs}")
        print("Empty_ctr: {} out of {} queries".format(empty_ctr, (en - st)))
        out_file = os.path.join(args.output_dir, "{}_train_chains_{}.pkl".format(args.dataset_name.lower(), str(args.job_id)))
        print("Writing file at {}".format(out_file))
        with open(out_file, "wb") as fout:
            pickle.dump(all_subgraphs, fout)
