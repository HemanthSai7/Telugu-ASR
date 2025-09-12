from src.dataset import get, get_shape
from src.helpers import (
    prepare_featurizers, 
    prepare_training_datasets, 
    prepare_training_dataloaders
)
from src.models import Conformer
from src.configs import Config
from src.losses import RnntLoss
from src.utils import env_util

from omegaconf import DictConfig, OmegaConf
from IPython.display import Audio

import os
import jiwer
import hydra
import tensorflow as tf

logger = tf.get_logger()


@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = None,
    spx: int = 1,
):
    config = Config(OmegaConf.to_container(config, resolve=True), training=True)
    
    tf.keras.backend.clear_session()
    env_util.setup_seed()
    strategy = env_util.setup_strategy(config.learning_config["running_config"]["devices"])
    batch_size = batch_size or config.learning_config["running_config"]["batch_size"]
    
    speech_featurizer, tokenizer = prepare_featurizers(config)

    train_dataset, valid_dataset = prepare_training_datasets(
        config,
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
    )

    shapes = get_shape(
        config,
        train_dataset,
        valid_dataset,
    )

    train_data_loader, valid_data_loader, global_batch_size = prepare_training_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        strategy=strategy,
        global_batch_size=batch_size,
        shapes=shapes,
    )

    print(shapes)

    for batch in train_data_loader:
        # print("Batch input keys:", batch[0].keys())
        # print("Batch target keys:", batch[1].keys())
        print("Prediction:", batch[0]["prediction"])
        print("Labels:", batch[1]["labels"])
        print("Input sentence",tokenizer.batch_decode(batch[0]["prediction"].numpy().tolist()))
        break

    with strategy.scope():
        model = Conformer(**config.model_config, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
        model.make(
            audio_input_shape=shapes["audio_input_shape"],
            audio_input_length_shape=shapes["audio_input_length_shape"],
            prediction_shape=shapes["prediction_shape"],
            prediction_length_shape=shapes["prediction_length_shape"],
            batch_size=global_batch_size
        )
        # model.run_eagerly = True
        model.summary(expand_nested=True)

        if config.learning_config["pretrained"]:
            model.load_weights(config.learning_config["pretrained"], by_name=True)
        model.compile(
            optimizer=tf.keras.optimizers.get(config.learning_config["optimizer_config"]),
            loss=RnntLoss(blank=0, global_batch_size=global_batch_size),
            run_eagerly=True,
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config["running_config"]["checkpoint"], verbose=1),
        tf.keras.callbacks.BackupAndRestore(config.learning_config["running_config"]["states_dir"]),
        tf.keras.callbacks.TensorBoard(**config.learning_config["running_config"]["tensorboard"]),
        tf.keras.callbacks.CSVLogger(config.learning_config["running_config"]["csv_logger"]),
    ]
    # print(model.count_params())

    # model.fit(
    #     train_data_loader,
    #     epochs=config.learning_config["running_config"]["num_epochs"],
    #     validation_data=valid_data_loader,
    #     steps_per_epoch=train_dataset.total_steps,
    #     validation_steps=valid_dataset.total_steps if valid_data_loader else None,
    #     callbacks=callbacks,
    #     verbose=1,
    # )

if __name__ == "__main__":
    main()