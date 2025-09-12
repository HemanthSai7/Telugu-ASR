def TrainInput(audio_inputs, audio_inputs_length, prediction, prediction_length):
    return {
        "audio_inputs": audio_inputs,
        "audio_inputs_length": audio_inputs_length,
        "prediction": prediction,
        "prediction_length": prediction_length
    }

def TargetLabels(labels, labels_length):
    return {
        "labels": labels,
        "labels_length": labels_length
    }

def OutputLogits(logits, logits_length):
    return {
        "logits": logits,
        "logits_length": logits_length
    }