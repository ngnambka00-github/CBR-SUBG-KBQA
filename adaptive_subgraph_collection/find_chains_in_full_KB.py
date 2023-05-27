import os
import pickle
import argparse
from collections import defaultdict

from tqdm import tqdm
from numpy.random import default_rng

from adaptive_subgraph_collection.adaptive_utils import get_query_entities_and_answers_metaqa, \
    read_metaqa_kb, find_paths

rng = default_rng()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities using CBR")
    parser.add_argument("--train_file", type=str, default='./data/train.json')
    parser.add_argument("--dataset_name", type=str, default='metaqa')
    parser.add_argument("--output_dir", type=str, default='./data/subgraph/')
    parser.add_argument("--kb_file", type=str, default="./data/kb.txt")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=1)
    args = parser.parse_args()

    # Read all entities and answers in MetaQA
    qid2qents, qid2answers, _, _ = get_query_entities_and_answers_metaqa(args.train_file)

    # MetaQA has its own KB and not full Freebase, hence do not need SPARQL
    # Read MetaQA KB
    # find 1, 2, 3 hop paths between question entities and answers
    all_subgraphs = defaultdict(list)
    e1_map = read_metaqa_kb(args.kb_file)
    qid2qents = [(qid, q_ents) for (qid, q_ents) in sorted(qid2qents.items(), key=lambda item: item[0])]
    job_size = len(qid2qents) / args.total_jobs
    st = args.job_id * job_size
    en = (1 + args.job_id) * job_size
    print("St: {}, En: {}".format(st, en))
    empty_ctr = 0
    all_len = []
    for ctr, (qid, q_ents) in tqdm(enumerate(qid2qents), total=len(qid2qents)):
        if st <= ctr < en:
            ans_ents = qid2answers[qid]
            len_q = 0
            for q_ent in q_ents:
                for ans_ent in ans_ents:
                    paths = find_paths(e1_map, q_ent, ans_ent)
                    all_subgraphs[qid].append({'st': q_ent, 'en': ans_ent, 'chains': paths})
                    len_q += len(paths)
                    if len(paths) == 0:
                        print(f"CHECK: q_ent: {q_ent} | ans_ent: {ans_ent}")
            if len_q == 0:
                empty_ctr += 1
            all_len.append(len_q)

    print(f"CHECK: {all_subgraphs}")
    print("Empty_ctr: {} out of {} queries".format(empty_ctr, (en - st)))
    out_file = os.path.join(args.output_dir,
                            "{}_train_chains_{}.pkl".format(args.dataset_name.lower(), str(args.job_id)))
    print("Writing file at {}".format(out_file))
    with open(out_file, "wb") as f_out:
        pickle.dump(all_subgraphs, f_out)
