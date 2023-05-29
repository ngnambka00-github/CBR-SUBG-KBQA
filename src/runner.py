import datetime
import json
import logging
import os
import sys
import uuid

import torch
from transformers import HfArgumentParser

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(path, os.pardir)))

from data_loaders.kbqa_dataloader import KBQADataLoader
from models.rgcn.rgcn_model import RGCN, QueryAwareRGCN
from models.compgcn.compgcn_models import CompGCN_TransE
from text_handler import PrecomputedQueryEncoder, QueryEncoder
from model_trainer import ModelTrainer
from global_config import logger
from data_loaders.training_utils import *
from run_arguments import ModelArguments, DataTrainingArguments, CBRTrainingArguments

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CBRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.transform_input = (model_args.transform_input == 1)
    if model_args.use_scoring_head == "none":
        model_args.use_scoring_head = None

    training_args.load_best_model_at_end = True
    if training_args.task == 'pt_match':
        project_tags = ["pt_match", "rgcn"]
        if data_args.dataset_name != 'synthetic':
            project_tags.append("kbqa")
    elif training_args.task == 'kbc':
        if model_args.gnn == 'CompGCN_TransE':
            project_tags = ['kbc', 'CompGCN_TransE']
        if model_args.gnn == 'RGCN':
            project_tags = ['kbc', 'RGCN']

    # making file output model training
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
    rand_str = str(uuid.uuid4())[:8]
    training_args.output_dir = os.path.join(training_args.output_dir, "out-" + suffix + rand_str)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # also log to a log file
    fileHandler = logging.FileHandler("{0}/{1}".format(training_args.output_dir, "log.txt"))
    logger.addHandler(fileHandler)
    logger.info("Output directory is {}".format(training_args.output_dir))
    logger.info("=========Config:============")
    logger.info(json.dumps(training_args.to_dict(), indent=4, sort_keys=True))
    logger.info(json.dumps(vars(model_args), indent=4, sort_keys=True))
    logger.info(json.dumps(vars(data_args), indent=4, sort_keys=True))
    logger.info("============================")
    if training_args.max_steps > 0:
        logger.info("max_steps is given, train will run till whichever is sooner of num_train_epochs and max_steps")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if training_args.task in ['pt_match', 'kbc']:
        dataset_obj = KBQADataLoader(data_args.data_dir, data_args.data_file_suffix, training_args.train_batch_size,
                                     training_args.eval_batch_size, model_args.add_dist_feature,
                                     model_args.add_inv_edges_to_edge_index, data_args.max_dist,
                                     training_args.downsample_eval_frac, training_args.task, data_args.dataset_name,
                                     data_args.precomputed_query_encoding_dir, data_args.paths_file_kbc,
                                     data_args.kb_system_file)
    else:
        raise NotImplemented(f"training_args.task: {training_args.task}")

    query_encoder = None
    model_args.node_feat_dim = dataset_obj.node_feat_dim
    model_args.n_additional_feat = dataset_obj.n_additional_feat
    model_args.n_base_feat = dataset_obj.n_base_feat
    model_args.max_dist = data_args.max_dist
    model_args.num_relations = dataset_obj.n_relations
    model_args.device = device
    if training_args.task == 'pt_match' and model_args.use_query_aware_gcn:
        if data_args.precomputed_query_encoding_dir is not None and training_args.train_query_encoder == 0:
            logger.info("query_encoder: using precomputed query encodings")
            query_encoder = PrecomputedQueryEncoder(dataset_obj)
        else:
            logger.info("query_encoder: creating query encoder model")
            if data_args.precomputed_query_encoding_dir is not None:
                logger.warning("query_encoder: ignoring precomputed query encodings")
            query_encoder = QueryEncoder(model_args.query_encoder_model, model_args.pooling_type,
                                         (training_args.train_query_encoder == 1)).to(device)
        # Set the query encoding dimension based on the chosen encoder
        model_args.query_dim = query_encoder.get_query_embedding_dim()
        if model_args.use_sparse_feats:
            solver_model = QueryAwareRGCN(model_args, dataset_obj.base_feature_matrix).to(device)
        else:
            solver_model = QueryAwareRGCN(model_args).to(device)
    else:
        if model_args.use_sparse_feats:
            solver_model = RGCN(model_args, dataset_obj.base_feature_matrix).to(device)
        else:
            solver_model = RGCN(model_args).to(device)

    if model_args.model_ckpt_path is not None and os.path.exists(model_args.model_ckpt_path):
        logger.info("Path to a checkpoint found; loading the checkpoint!!!")
        state_dict = torch.load(model_args.model_ckpt_path)
        solver_model.load_state_dict(state_dict)
    optim_state_dict = None
    if model_args.optim_ckpt_path is not None and os.path.exists(model_args.optim_ckpt_path):
        logger.info("Path to a OPTIMIZER checkpoint found; loading the checkpoint!!!")
        optim_state_dict = torch.load(model_args.optim_ckpt_path)
    global_step = None
    if model_args.model_args_ckpt_path is not None and os.path.exists(model_args.model_args_ckpt_path):
        logger.info("Path to a model_args checkpoint found; loading the global_step!!!")
        with open(model_args.model_args_ckpt_path) as fin:
            loaded_model_args = json.load(fin)
            # load the global step
            global_step = loaded_model_args["global_step"]

    if training_args.patience:
        early_stopping = EarlyStopping("Hits@1", patience=training_args.patience)
    else:
        early_stopping = None

    trainer = ModelTrainer(solver_model, query_encoder, dataset_obj, training_args=training_args, data_args=data_args,
                           model_args=model_args, optim_state_dict=optim_state_dict, global_step=global_step,
                           device=device, early_stopping=early_stopping)
    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        if training_args.do_train:
            logger.warning("Evaluating current trained model...")
        elif model_args.model_ckpt_path is None or not os.path.exists(model_args.model_ckpt_path):
            logger.warning("No path to model found!!!, Evaluating with a random model...")
        trainer.evaluate(log_output=(training_args.log_eval_result == 1))

    if training_args.do_predict:
        # predict all batches
        # if model_args.model_ckpt_path is None or not os.path.exists(model_args.model_ckpt_path):
        #     logger.warning("No path to model found!!!, Evaluating with a random model...")
        # trainer.predict()

        # predict_single
        pred = trainer.single_predict()
        print(f"Top 5 Answers: {pred[0][:5]}")
