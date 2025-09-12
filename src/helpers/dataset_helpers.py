from src.speech_featurizer import SpeechFeaturizer
from src.tokenizer import extract_telugu_tokens_from_pretrained, TeluguTokenizer
from src.dataset import get

import os
import string
import tensorflow as tf

logger = tf.get_logger()

def debug_tokenizer(tokenizer, sample_tokens=[185, 49, 218, 237, 240, 237, 166, 58, 165, 102]):
    """Debug tokenizer vocabulary and decoding using batch_decode."""
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens:")
    print(f"  BOS: {tokenizer.bos_token_id} -> '{tokenizer.bos_token}'")
    print(f"  EOS: {tokenizer.eos_token_id} -> '{tokenizer.eos_token}'")  
    print(f"  PAD: {tokenizer.pad_token_id} -> '{tokenizer.pad_token}'")
    print(f"  UNK: {tokenizer.unk_token_id} -> '{tokenizer.unk_token}'")
    
    print(f"\nVocabulary sample (first 20 tokens):")
    for i in range(min(200, tokenizer.vocab_size)):
        token = tokenizer._convert_id_to_token(i)
        print(f"  {i}: '{token}'")
    
    print(f"\nHigh-ID tokens that are being generated:")
    for token_id in sample_tokens[:10]:
        if token_id < tokenizer.vocab_size:
            token = tokenizer._convert_id_to_token(token_id)
            print(f"  {token_id}: '{token}'")
        else:
            print(f"  {token_id}: OUT_OF_VOCAB")
    
    print(f"\nTesting batch_decode of sample sequence:")
    try:
        decoded_batch = tokenizer.batch_decode([sample_tokens], skip_special_tokens=False)
        print(f"  Batch Decoded: '{decoded_batch[0]}'")
    except Exception as e:
        print(f"  Batch decode error: {e}")
    
    print(f"\nTesting batch_decode for individual tokens:")
    try:
        # Each token_id as a single-token sequence
        decoded_individual = tokenizer.batch_decode([[tid] for tid in sample_tokens[:10]], skip_special_tokens=True)
        for token_id, token_str in zip(sample_tokens[:10], decoded_individual):
            print(f"  {token_id} -> '{token_str}'")
        print(f"\nManual concatenation: '{''.join(decoded_individual)}'")
    except Exception as e:
        print(f"  Individual batch decode error: {e}")
    
    
    ground_truth = "కరెంటు బిల్లుల్ని కూడా సులువుగా ఆండ్రాయిడ్ ఫోన్ ఉంటే జీపే ద్వారా చెయ్యొచ్చు"
    print(f"\nTesting ground truth encoding/decoding:")
    print(f"  Original: '{ground_truth}'")
    try:
        encoded = tokenizer.encode(ground_truth)
        print(f"  Encoded: {encoded}")
        decoded_back = tokenizer.batch_decode([encoded], skip_special_tokens=True)[0]
        print(f"  Decoded back: '{decoded_back}'")
        print(f"  Round-trip successful: {ground_truth == decoded_back}")
    except Exception as e:
        print(f"  Encoding/decoding error: {e}")

def prepare_featurizers(
    config,
):
    speech_config = config.speech_config
    feature_extractor = SpeechFeaturizer(**dict(speech_config))

    # chars = string.ascii_lowercase + " " # This is character vocab
    # tokenizer = CharacterTokenizer(chars, model_max_length=None)

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.getenv("HF_TOKEN"))
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_name = "google/gemma-2-2b"
    telugu_tokens = extract_telugu_tokens_from_pretrained(model_name)
    telugu_tokens.extend([" ","."])
    print(f"Telugu tokens: {telugu_tokens} with length: {len(telugu_tokens)}")

    tokenizer = TeluguTokenizer(telugu_tokens, model_max_length=None)
    # debug_tokenizer(tokenizer)
    # ff
    
    return feature_extractor, tokenizer

def prepare_training_datasets(
    config,
    speech_featurizer: SpeechFeaturizer,
    tokenizer: TeluguTokenizer,
):
    train_dataset = get(
        tokenizer=tokenizer,
        speech_featurizer=speech_featurizer,
        dataset_config=config.data_config["train_dataset_config"],
    )
    valid_dataset = get(
        tokenizer=tokenizer,
        speech_featurizer=speech_featurizer,
        dataset_config=config.data_config["eval_dataset_config"],
    )

    return train_dataset, valid_dataset

def prepare_training_dataloaders(
    train_dataset,
    valid_dataset,
    strategy,
    global_batch_size,
    shapes,
):
    global_batch_size *= strategy.num_replicas_in_sync
    train_data_loader = train_dataset.create(batch_size=global_batch_size, padded_shapes=shapes)
    valid_data_loader = valid_dataset.create(batch_size=global_batch_size, padded_shapes=shapes)

    return train_data_loader, valid_data_loader, global_batch_size

def prepare_testing_datasets(
    config,
    speech_featurizer: SpeechFeaturizer,
    tokenizer: TeluguTokenizer,
):
    test_dataset = get(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_config=config.data_config["test_dataset_config"],
    )
    
    return test_dataset

def prepare_testing_dataloaders(
    test_dataset,
    strategy,
    global_batch_size,
    shapes,
):
    global_batch_size *= strategy.num_replicas_in_sync
    test_data_loader = test_dataset.create(batch_size=global_batch_size, padded_shapes=shapes)

    return test_data_loader, global_batch_size

