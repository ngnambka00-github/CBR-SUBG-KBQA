import json
import os
from tqdm import tqdm
import pickle
import argparse

from adaptive_subgraph_collection.adaptive_utils import get_query_entities_and_answers_metaqa


def read_cbr_subgraphs(subgraph_train_file, subgraph_dev_file=None, subgraph_test_file=None):
    cbr_subgraph = {}
    if os.path.exists(subgraph_train_file):
        with open(subgraph_train_file, "rb") as fin:
            cbr_subgraph.update(pickle.load(fin))
    if os.path.exists(subgraph_dev_file):
        with open(subgraph_dev_file, "rb") as fin:
            cbr_subgraph.update(pickle.load(fin))
    if os.path.exists(subgraph_test_file):
        with open(subgraph_test_file, "rb") as fin:
            cbr_subgraph.update(pickle.load(fin))
    new_subgraphs = {}
    replace_ctr = 0
    for ctr, (qid, triples) in tqdm(enumerate(cbr_subgraph.items())):
        new_triples = []
        for (e1, r, e2) in triples:
            if e1.endswith('-08:00'):
                e1 = e1[:-6]
                replace_ctr += 1
            if e2.endswith('-08:00'):
                e2 = e2[:-6]
                replace_ctr += 1
            new_triples.append((e1, r, e2))
        assert len(new_triples) == len(triples)
        new_subgraphs[qid] = new_triples
    print(replace_ctr)
    assert len(new_subgraphs) == len(cbr_subgraph)
    return new_subgraphs


def create_new_vocab(args, cbr_subgraph, qid2answers, output_dir):
    all_entities, all_relations = set(), set()
    for qid, subgraph in tqdm(cbr_subgraph.items()):
        for (e1, r, e2) in subgraph:
            all_entities.add(e1)
            all_entities.add(e2)
            all_relations.add(r)
    # add the answers to this list as well, in case the subgraph dont cover answers
    for _, answers in qid2answers.items():
        for ans in answers:
            all_entities.add(ans)
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = {}, {}, {}, {}

    for e_ctr, e in enumerate(all_entities):
        entity_vocab[e] = e_ctr
        rev_entity_vocab[e_ctr] = e
    for r_ctr, r in enumerate(all_relations):
        rel_vocab[r] = r_ctr
        rev_rel_vocab[r_ctr] = r

    # write vocab
    new_ent_vocab_file = os.path.join(output_dir, f"entities_roberta-base_mean_pool_masked_cbr_subgraph_k={args.k}.txt")
    new_rel_vocab_file = os.path.join(output_dir,
                                      f"relations_roberta-base_mean_pool_masked_cbr_subgraph_k={args.k}.txt")
    print("Writing new vocab files: {}, {}".format(new_ent_vocab_file, new_rel_vocab_file))
    with open(new_ent_vocab_file, "w") as f_out:
        for i in range(len(entity_vocab)):
            f_out.write(rev_entity_vocab[i] + "\n")
    with open(new_rel_vocab_file, "w") as f_out:
        for i in range(len(rel_vocab)):
            f_out.write(rev_rel_vocab[i] + "\n")

    return entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab


# TODO: rewrite the function so that it is independent of the KNN file. Right now, most of the
# TODO: input is copied from the existing file (containing the KNNs)
def write_files_with_new_subgraphs(input_data, output_file, cbr_subgraph, ent_vocab, rel_vocab, qid2q_ents):
    output_data = []
    for d in tqdm(input_data):
        qid = d["id"]
        new_subgraph_test = set()
        new_subgraph = list(cbr_subgraph[qid])
        all_entities = set()
        for (e1, r, e2) in new_subgraph:
            new_subgraph_test.add((ent_vocab[e1], rel_vocab[r], ent_vocab[e2]))
            all_entities.add(ent_vocab[e1])
            all_entities.add(ent_vocab[e2])
        d["subgraph"] = {}
        d["subgraph"]["tuples"] = list(new_subgraph_test)
        d["subgraph"]["entities"] = list(all_entities)
        # fill in seed entities
        q_ents = qid2q_ents[qid]
        seed_ents = []
        for e in q_ents:
            if e in ent_vocab:
                seed_ents.append(ent_vocab[e])
        d["seed_entities"] = seed_ents
        output_data.append(d)
    print("Writing data to {}".format(output_file))
    with open(output_file, "w") as fout:
        json.dump(output_data, fout, indent=2)
    print("Done...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create input files using CBR subgraphs")
    parser.add_argument("--subgraph_dir", type=str, default="./data/subgraph")
    parser.add_argument("--input_dir", type=str, default="./data/")
    parser.add_argument("--dataset_name", type=str, default="metaqa")
    parser.add_argument("--output_dir", type=str, default="./data/subgraph")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    # build path
    args.subgraph_train_file = f"{args.dataset_name}_cbr_subgraph_train_{args.k}.pkl"
    args.subgraph_dev_file = f"{args.dataset_name}_cbr_subgraph_dev_{args.k}.pkl"
    args.subgraph_test_file = f"{args.dataset_name}_cbr_subgraph_test_{args.k}.pkl"
    subgraph_train_file = os.path.join(args.subgraph_dir, args.subgraph_train_file)
    subgraph_dev_file = os.path.join(args.subgraph_dir, args.subgraph_dev_file)
    subgraph_test_file = os.path.join(args.subgraph_dir, args.subgraph_test_file)

    print("Reading CBR subgraphs ...")
    cbr_subgraph = read_cbr_subgraphs(subgraph_train_file, subgraph_dev_file, subgraph_test_file)

    print("Getting query entities and answers....")
    train_qid2qents, train_qid2answers, _, train_qid2q_str = None, None, None, None
    train_file = os.path.join(args.input_dir, "train.json")
    dev_file = os.path.join(args.input_dir, "dev.json")
    test_file = os.path.join(args.input_dir, "test.json")
    if os.path.exists(train_file):
        train_qid2qents, train_qid2answers, _, train_qid2q_str = get_query_entities_and_answers_metaqa(train_file)
    dev_qid2qents, dev_qid2answers, dev_qid2gold_chains, dev_qid2q_str = None, None, None, None
    if os.path.exists(dev_file):
        dev_qid2qents, dev_qid2answers, _, dev_qid2q_str = get_query_entities_and_answers_metaqa(dev_file)
    test_qid2qents, test_qid2answers, test_qid2gold_chains, test_qid2q_str = None, None, None, None
    if os.path.exists(test_file):
        test_qid2qents, test_qid2answers, _, test_qid2q_str = get_query_entities_and_answers_metaqa(test_file)

    print("Creating new vocab....")
    qid2answers = {}
    if train_qid2answers is not None:
        qid2answers.update(train_qid2answers)
    if dev_qid2answers is not None:
        qid2answers.update(dev_qid2answers)
    if test_qid2answers is not None:
        qid2answers.update(test_qid2answers)
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_new_vocab(args, cbr_subgraph, qid2answers,
                                                                                args.output_dir)
    print("Reading input files")
    train_file_name = os.path.join(args.input_dir, "train.json")
    test_file_name = os.path.join(args.input_dir, "test.json")
    dev_file_name = os.path.join(args.input_dir, "dev.json")
    with open(train_file_name) as fin:
        train = json.load(fin)
    with open(test_file_name) as fin:
        test = json.load(fin)
    with open(dev_file_name) as fin:
        dev = json.load(fin)

    qid2q_ents = {}
    for qe in [train_qid2qents, dev_qid2qents, test_qid2qents]:
        if qe is not None:
            qid2q_ents.update(qe)

    print("Creating files with new subgraphs...")
    print("Train...")
    output_file = os.path.join(args.output_dir, f"train_roberta-base_mean_pool_masked_cbr_subgraph_k={args.k}.json")
    write_files_with_new_subgraphs(train, output_file, cbr_subgraph, entity_vocab, rel_vocab, qid2q_ents)
    print(f"File written to {output_file}")

    print("Dev...")
    output_file = os.path.join(args.output_dir, f"dev_roberta-base_mean_pool_masked_cbr_subgraph_k={args.k}.json")
    write_files_with_new_subgraphs(dev, output_file, cbr_subgraph, entity_vocab, rel_vocab, qid2q_ents)
    print(f"File written to {output_file}")

    print("Test...")
    output_file = os.path.join(args.output_dir, f"test_roberta-base_mean_pool_masked_cbr_subgraph_k={args.k}.json")
    write_files_with_new_subgraphs(test, output_file, cbr_subgraph, entity_vocab, rel_vocab, qid2q_ents)
    print(f"File written to {output_file}")
