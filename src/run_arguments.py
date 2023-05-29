from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CBRTrainingArguments(TrainingArguments):
    """
    subclass of HF training arguments.
    """
    task: str = field(default='pt_match', metadata={"help": "Options: [kbc, pt_match]"})
    dist_metric: str = field(default='l2', metadata={"help": "Options for pt_match: [l2, cosine], "
                                                             "Currently no options for kbc"})
    dist_aggr1: str = field(default='mean', metadata={"help": "Distance aggregation function at each neighbor query. "
                                                              "Options: [none (no aggr), mean, sum]"})
    dist_aggr2: str = field(default='mean', metadata={"help": "Distance aggregation function across all neighbor "
                                                              "queries. Options: [mean, sum]"})
    loss_metric: str = field(default='margin', metadata={"help": "Options for pt_match: [margin, txent], "
                                                                 "Options for kbc: [bce, dist]"})
    margin: float = field(default=5.0, metadata={"help": "Margin for loss computation"})
    sampling: float = field(default=1.0, metadata={"help": "Fraction of negative samples used"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for temperature scaled cross-entropy loss"})
    log_eval_result: int = field(default=0, metadata={"help": "Whether to log distances and ranking during evaluation"})
    train_batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    eval_batch_size: int = (field(default=8, metadata={"help": "Evaluation batch size"}))
    learning_rate: float = field(default=0.001, metadata={"help": "Starting learning rate"})
    train_query_encoder: int = field(default=0, metadata={"help": "Whether to train the query encoder model when "
                                                                  "training query-aware message passing networks"})
    encoder_learning_rate: float = field(default=5e-5, metadata={"help": "Initial learning rate for query encoder."})
    warmup_steps: int = (field(default=0, metadata={"help": "scheduler warm up steps"}))
    downsample_eval_frac: float = field(default=1.0, metadata={"help": "Fraction of dev set to use for evaluation. "
                                                                       "Currently only implemented for pt_match"})
    kbc_eval_type: str = field(default='both', metadata={"help": "head/tail/both"})
    patience: int = field(default=None, metadata={"help": "Early Stopping Patience"})


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(metadata={"help": "synthetic is a special dataset. all other datasets are treated as "
                                                "kb completion datasets"})
    data_dir: str = field(metadata={"help": "The path to data directory (contains train.json, dev.json, test.json)."})
    data_file_suffix: str = field(default='roberta-base_mean_pool_masked_cbr_subgraph_k=10',
                                  metadata={"help": "The suffix s for using train_s.json, dev_s.json, "
                                                    "test_s.json instead of train.json, dev.json, "
                                                    "test.json."})
    kb_system_file: str = field(default=None, metadata={
        "help": "The path to KB system file containing the full list of relations."})
    precomputed_query_encoding_dir: str = field(default=None, metadata={
        "help": "The path to directory containing precomputed query encodings query_enc_{train,dev,test}.pt. "
                "Will raise an error if used with train_query_encoder=1"})
    max_dist: int = field(default=3, metadata={"help": "When using distance from seed node as feature, this is the "
                                                       "maximum distance expected (would be the radius of the graph "
                                                       "from seed entities)"})
    otf: bool = field(default=False,
                      metadata={"help": "Use on the fly subgraph sampling, otherwise load paths from pkl file"})
    otf_max_nodes: int = field(default=1000,
                               metadata={"help": "Maximum number of nodes per subgraph in on-the-fly sampling"})
    edge_dropout: float = field(default=0.0, metadata={"help": "Percentage of edges in subgraphs to randomly remove"})
    node_dropout: float = field(default=0.0, metadata={"help": "Percentage of nodes in subgraphs to randomly remove"})
    num_neighbors_train: int = field(default=1,
                                     metadata={
                                         "help": "Number of near-neighbor subgraphs, k, to train with. K number of graphs will be randomly sampled from a larger list"})
    num_neighbors_eval: int = field(default=5,
                                    metadata={"help": "Number of near-neighbor subgraphs, k, to eval with"})
    adaptive_subgraph_k: int = field(default=25,
                                     metadata={
                                         "help": "Number of nearest neighbors used for creating the subgraphs for each question."})
    label_smooth: float = field(default=0.0, metadata={"help": "label smoothing"})
    paths_file_kbc: str = field(default='paths_1000_len_3.pkl', metadata={"help": "Paths file name"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    transform_input: int = field(default=0, metadata={"help": "Add linear transform over one-hot input encoding"})
    use_fast_rgcn: bool = field(default=True, metadata={"help": "Choose between RGCNConv (GPU memory-efficient by"
                                                                " iterating over each individual relation type) and"
                                                                " FastRGCNConv"})
    use_query_aware_gcn: int = field(default=0, metadata={"help": "Choose between vanilla RGCN and question aware "
                                                                  "variation (only used by KBQA)"})
    transform_query: int = field(default=0, metadata={"help": "Add linear transform over query encoding"})
    query_proj_dim: int = field(default=32, metadata={"help": "When using transform_query, dim to project down to"})
    query_attn_type: str = field(default=None, metadata={"help": "Type of query-aware attention to implement. "
                                                                 "Options: ['full', 'dim', 'sep']"})
    query_attn_activation: str = field(default='softmax', metadata={"help": "Activation fn for query-aware attention. "
                                                                            "Options: ['softmax', 'sigmoid']"})
    query_encoder_model: str = field(default=None, metadata={"help": "Model card or ckpt path compatible with the"
                                                                     " transformers library. [Tested for "
                                                                     "`roberta-base`]"})
    pooling_type: str = field(default='pooler', metadata={"help": "Output pooling to use for query encoding. "
                                                                  "Options: ['pooler', 'cls', 'mean_pool']"})
    node_feat_dim: int = field(default=None, metadata={"help": "Dimension of node input features"})
    dense_node_feat_dim: int = field(default=512, metadata={
        "help": "If not using sparse features, dimension of input entity embedding"})
    use_sparse_feats: bool = field(default=True, metadata={"help": "1 if using sparse_feats"})
    gcn_dim: int = field(default=32, metadata={"help": "GCN layer dimension"})
    num_bases: int = field(default=None, metadata={"help": "Number of bases for basis-decomposition of relation "
                                                           "embeddings"})
    num_gcn_layers: int = field(default=3, metadata={"help": "Number of GCN layers"})
    add_dist_feature: bool = field(default=True,
                                   metadata={"help": "Add (one-hot) distance from seed node as feature to "
                                                     "entity repr"})
    add_inv_edges_to_edge_index: bool = field(default=True,
                                              metadata={"help": "[SYNTHETIC DATA] Include inverse relations "
                                                                "in message passing. By default, messages are"
                                                                " only passed one way"})
    use_scoring_head: str = field(default=None, metadata={"help": "Options: [transe, none]"})
    model_ckpt_path: str = field(default=None, metadata={"help": "Checkpoint to load"})
    optim_ckpt_path: str = field(default=None, metadata={"help": "Optimizer checkpoint to load"})
    model_args_ckpt_path: str = field(default=None, metadata={"help": "Model args to load"})
    gnn: str = field(default="RGCN", metadata={"help": "Which GNN model to use on subgraphs"})
    drop_rgcn: float = field(default=0.0, metadata={"help": "Dropout probability for RGCN model"})
