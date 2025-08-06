import os
import random
import numpy as np
import torch
import logging
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['FLAGS_cudnn_deterministic'] = "True"

def log_args(args):
    s = "\n==========================================\n"
    
    s += ("python " + " ".join(sys.argv) + "\n")
    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    
    s += "==========================================\n"
    
    return s

def set_logger(args, log_name="train_log.txt"):
    
    log_format = "%(asctime)s [%(levelname)s] - %(message)s"
    # creating logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="w")
    file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    
    # terminal logger handler
    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler) 
    # if not args.test:
    logger.debug(log_args(args))
    
    return logger
