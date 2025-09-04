from src.metrics.asr_metrics import ErrorRate
from src.utils.file_util import read_file
from src.utils import wer, cer

import tensorflow as tf
from tqdm import tqdm

logger = tf.get_logger()


def evaluate_results(
        filepath: str,
):
    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "greedy_wer": ErrorRate(wer, name="greedy_wer", dtype=tf.float32),
        "greedy_cer": ErrorRate(cer, name="greedy_cer", dtype=tf.float32),
    }

    with read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
            lines = lines[1:] # skip header

        for eachline in tqdm(lines):
            _, _, groundtruth, greedy = eachline.split("\t")
            
            
            groundtruth = [groundtruth]
            greedy = [greedy]
            metrics["greedy_wer"].update_state(groundtruth, greedy)
            metrics["greedy_cer"].update_state(groundtruth, greedy)
        for key, value in metrics.items():
            logger.info(f"{key}: {value.result().numpy()}")