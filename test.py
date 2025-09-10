from src.dataset import get_shape
from src.helpers import (
    prepare_featurizers, 
    prepare_testing_datasets, 
    prepare_testing_dataloaders, 
    run_testing
)
from src.configs import Config
from src.utils import env_util
from src.model import ASRModel

from omegaconf import DictConfig, OmegaConf

import hydra
import tensorflow as tf

logger = tf.get_logger()


@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = None,
    saved: str = "../../../logs/model/2025-09-06/checkpoints/35.h5", #28, 24,14,17 best | 29, 25, 26, 27 okayish
    output: str = "output.tsv",
):
    config = Config(OmegaConf.to_container(config, resolve=True), training=True)
    
    tf.keras.backend.clear_session()
    env_util.setup_seed()
    strategy = env_util.setup_strategy(config.learning_config["running_config"]["devices"])
    batch_size = batch_size or config.learning_config["running_config"]["batch_size"]
    
    speech_featurizer, tokenizer = prepare_featurizers(config)

    test_dataset = prepare_testing_datasets(
        config,
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
    )

    shapes = get_shape(
        config,
        test_dataset,
    )

    test_data_loader, global_batch_size = prepare_testing_dataloaders(
        test_dataset,
        strategy,
        batch_size,
        shapes,
    )

    model = ASRModel(**config.model_config, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
    model.make(**shapes, batch_size=global_batch_size)
    # model.run_eagerly = False
    model.load_weights(saved, skip_mismatch=False, by_name=True)
    model.summary()

    print(test_dataset.total_steps)

    run_testing(
        model=model,
        test_dataset=test_dataset,
        test_data_loader=test_data_loader,
        output=output,
    )

if __name__ == "__main__":
    main()