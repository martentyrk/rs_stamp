import argparse
from logging import getLogger
from pytorch_implementation.utils import init_logger, init_seed, get_model, get_trainer, set_color
from pytorch_implementation.config import Config
from pytorch_implementation.model.stamp import STAMP

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="diginetica-session",
        help="Benchmarks for session-based rec.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Whether evaluating on validation set (split from train set), otherwise on test set.",
    )
    parser.add_argument(
        "--valid_portion", type=float, default=0.1, help="ratio of validation set."
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()
    
    
    config_dict = {
        "USER_ID_FIELD": "session_id",
        "load_col": None,
        "neg_sampling": None,
        "benchmark_filename": ["train", "test"],
        "alias_of_item_id": ["item_id_list"],
        "topk": [20],
        "metrics": ["Recall", "MRR"],
        "valid_metric": "MRR@20",
    }
    
    config = Config(
        model="STAMP", dataset=f"{args.dataset}", config_dict=config_dict
    )
    
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)
    
    dataset = None
    
    model = STAMP(config=config, dataset=dataset)
    
    sequential_init = os.path.join(quick_start_config_path, "sequential.yaml")
    
    special_sequential_on_ml_100k_init = os.path.join(quick_start_config_path, "special_sequential_on_ml-100k.yaml")
    
    
    
    
    