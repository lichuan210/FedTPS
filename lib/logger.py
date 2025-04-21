import os
import logging


def get_logger(args, name=None):
    file_name = f"{args.dataset}_client_num_{args.num_client}_{args.mode}.log"
    logging_path = os.path.join(args.path, file_name)

    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    file_handler = logging.FileHandler(logging_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
