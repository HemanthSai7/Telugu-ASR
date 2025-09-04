from src.utils import file_util, app_util
from src.model import BaseModel
from src.dataset import ASRSliceDataset


from tqdm import tqdm
import tensorflow as tf

logger = tf.get_logger()


def run_testing(
    model: BaseModel,
    test_dataset: ASRSliceDataset,
    test_data_loader: tf.data.Dataset,
    output: str,
):
    with file_util.save_file(file_util.preprocess_paths(output)) as filepath:
        overwrite = True
        if tf.io.gfile.exists(filepath):
            overwrite = input(f"Overwrite existing result file {filepath} ? (y/n): ").lower() == "y"
        if overwrite:
            results = model.predict(test_data_loader, steps=test_dataset.total_steps, verbose=1)
            # results = tf.expand_dims(results, axis=0).numpy().tolist()
            # print(f"Resutls: {results}")
            logger.info(f"Saving result to {output} ...")
            with open(filepath, "w") as openfile:
                openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\n")
                progbar = tqdm(total=test_dataset.total_steps, unit="batch")
                for i, pred in enumerate(results):
                    # print(pred)
                    # print([x for x in pred])
                    groundtruth, greedy = [x.decode("utf-8") for x in pred]
                    path, duration, _ = test_dataset.entries[i]
                    openfile.write(f"{path}\t{duration}\t{groundtruth}\t{greedy}\n")
                    progbar.update(1)
                progbar.close()
        app_util.evaluate_results(filepath)


def convert_tflite(
    model: BaseModel,
    output: str,
    representative_dataset_gen,
    beam_width: int = 0,
):
    concrete_func = model.make_tflite_function(beam_width=beam_width).get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    output = file_util.preprocess_paths(output)
    with open(output, "wb") as tflite_out:
        tflite_out.write(tflite_model)
