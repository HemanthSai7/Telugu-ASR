def TrainInput(audio_inputs, shifted_right_text_inputs):
    return {
        "audio_inputs": audio_inputs,
        "shifted_right_text_inputs": shifted_right_text_inputs,
    }

def TargetLabels(text_targets):
    return {"text_targets": text_targets}