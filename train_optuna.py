from src.dataset import get, get_shape
from src.helpers import prepare_featurizers, prepare_training_datasets, prepare_training_dataloaders
from src.model import ASRModel
from src.configs import Config
from src.loss import MaskedCrossEntropyLoss
from src.utils import env_util
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.integration import TFKerasPruningCallback
from optuna_integration.wandb import WeightsAndBiasesCallback

from omegaconf import DictConfig, OmegaConf
from IPython.display import Audio

import os
import jiwer
import hydra
import optuna
import datetime
import tensorflow as tf

logger = tf.get_logger()

wandb_kwargs = {"project": "asr-hyperparam-snacks-telugu-15m"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

def train_with_optuna(config, trial):

    min_lr = trial.suggest_float('min_lr', 1e-6, 1e-5, log=True)
    max_lr = trial.suggest_float('max_lr', 1e-4, 5e-4, log=True)
    warmup_steps = trial.suggest_int('warmup_steps', 5000, 15000, log=True)

    opt_config = config.learning_config["optimizer_config"]["config"]
    opt_config["learning_rate"]["min_lr"] = min_lr
    opt_config["learning_rate"]["max_lr"] = max_lr
    opt_config["learning_rate"]["warmup_steps"] = warmup_steps

    # Regularization parameters
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4)

    config.model_config["kernel_regularizer"]["config"]["l2"] = weight_decay
    config.model_config["bias_regularizer"]["config"]["l2"] = weight_decay

    # Model architecture parameters
    d_model = trial.suggest_categorical('d_model', [192, 256, 320])
    config.model_config["d_model"] = d_model
    head_dim = trial.suggest_categorical('head_dim', [32, 48, 64])
    num_heads = d_model // head_dim

    if d_model % head_dim != 0:
        # Adjust head_dim to make it divisible
        head_dim = 64 if d_model >= 256 else 32
        num_heads = d_model // head_dim

    encoder_blocks = trial.suggest_int('encoder_blocks', 4, 8)
    decoder_blocks = trial.suggest_int('decoder_blocks', 2, 6)
    config.model_config["d_model"] = d_model

    for side, blocks in [("encoder_config", encoder_blocks), ("decoder_config", decoder_blocks)]:
        sub_cfg = config.model_config[side]
        sub_cfg["num_heads"] = num_heads
        sub_cfg["head_dim"] = head_dim
        sub_cfg["num_blocks"] = blocks
        
        # Dropout - can be lower with 95k files
        dropout_base = 0.1 if side == "encoder_config" else 0.05
        dropout_range = trial.suggest_float(f'{side}_dropout_rate', 
                                          dropout_base, dropout_base + 0.15)
        sub_cfg["dropout_rate"] = dropout_range

    # TF Setup
    tf.keras.backend.clear_session()
    env_util.setup_seed()
    strategy = env_util.setup_strategy(config.learning_config["running_config"]["devices"])
    batch_size = config.learning_config["running_config"]["batch_size"]

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

    with strategy.scope():
        model = ASRModel(**config.model_config, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
        model.make(**shapes, batch_size=global_batch_size)

        actual_params = model.count_params()
        trial.set_user_attr("actual_params", actual_params)
        trial.set_user_attr("actual_params_M", actual_params / 1e6)

        print(f"Trial {trial.number}: Actual parameters: {actual_params/1e6:.2f}M")
        
        if actual_params > 15_000_000:
            raise optuna.TrialPruned(f"Actual params {actual_params/1e6:.1f}M > 15M limit")
        model.summary(expand_nested=False)

        if config.learning_config["pretrained"]:
            model.load_weights(config.learning_config["pretrained"], by_name=True)
        
        optimizer = tf.keras.optimizers.get(config.learning_config["optimizer_config"])
        loss = MaskedCrossEntropyLoss(global_batch_size=global_batch_size, ignore_class=tokenizer.pad_token_id)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            run_eagerly=False,
        )

    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.learning_config["running_config"]["checkpoint"]["filepath"] = config.learning_config["running_config"]["checkpoint"]["filepath"].replace("logs/", f"logs/{log_dir}/")
    config.learning_config["running_config"]["states_dir"] = config.learning_config["running_config"]["states_dir"].replace("logs/", f"logs/{log_dir}/")
    config.learning_config["running_config"]["tensorboard"]["log_dir"] = config.learning_config["running_config"]["tensorboard"]["log_dir"].replace("logs/", f"logs/{log_dir}/")
    config.learning_config["running_config"]["csv_logger"] = config.learning_config["running_config"]["csv_logger"].replace("logs/", f"logs/{log_dir}/")

    callbacks = [
        TFKerasPruningCallback(trial, 'val_loss'),
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config["running_config"]["checkpoint"], verbose=1),
        tf.keras.callbacks.BackupAndRestore(config.learning_config["running_config"]["states_dir"]),
        tf.keras.callbacks.TensorBoard(**config.learning_config["running_config"]["tensorboard"]),
        tf.keras.callbacks.CSVLogger(config.learning_config["running_config"]["csv_logger"]),
    ]

    history = model.fit(
        train_data_loader,
        epochs= config.learning_config["running_config"]["num_epochs"],
        validation_data=valid_data_loader,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=valid_dataset.total_steps if valid_data_loader else None,
        callbacks=callbacks,
        verbose=1,
    )
    return min(history.history["val_loss"])

@wandbc.track_in_wandb()
def objective(trial):
    with hydra.initialize(config_path="config"):
        config = hydra.compose(config_name="config")
    
    config = Config(OmegaConf.to_container(config, resolve=True), training=True)
    return train_with_optuna(config, trial)

if __name__ == "__main__":
    sampler = TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)
    pruner = SuccessiveHalvingPruner(
        min_resource=8, 
        reduction_factor=3, 
        min_early_stopping_rate=4
    )

    storage_url = "sqlite:///asr_telugu_15m_hyperparam.db"
    study_name = "telugu_asr_15m_constraint"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize", 
        sampler=sampler, 
        pruner=pruner, 
        storage=storage_url, 
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=75, callbacks=[wandbc], timeout=None)

    print("="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Print parameter count info
    if "actual_params_M" in study.best_trial.user_attrs:
        print(f"Best model parameters: {study.best_trial.user_attrs['actual_params_M']:.2f}M")
    
    # Save best config for final training
    best_params_file = f"best_params_trial_{study.best_trial.number}.json"
    import json
    with open(best_params_file, 'w') as f:
        json.dump({
            'params': study.best_trial.params,
            'value': study.best_value,
            'user_attrs': study.best_trial.user_attrs
        }, f, indent=2)
    
    print(f"Best parameters saved to {best_params_file}")
