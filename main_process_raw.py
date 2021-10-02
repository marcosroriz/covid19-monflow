# Preâmbulo
import click
import csv
import glob
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
import tensorflow as tf
import time

from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Constants and setup info
matplotlib.use("Qt5Agg")


######################################################################################################################
# Baixa o modelo em questão
######################################################################################################################
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)


######################################################################################################################
# Baixa os labels
######################################################################################################################
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)


#######################################################################################################################
# Transforma uma imagem para um vetor np
#######################################################################################################################
def load_image_into_numpy_array(path):
    img = Image.open(path)
    return np.array(img), img.width, img.height


#######################################################################################################################
# Funções de detecção
#######################################################################################################################
@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


@click.command()
@click.option('--accuracy', type=float, default=0.2, help='accuracy threshold')
@click.option('--model', type=str, default='efficientdet_d4_coco17_tpu-32',
              help='model name (see https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md')
@click.option('--inputdir', type=str, default='C:/Imagens/RAW/', help='input folder')
@click.option('--outputdir', type=str, default='C:/Imagens/Output/RAW/', help='output folder')
@click.option('--use-gpu', is_flag=True, default=False, help='should we use gpu')
@click.option('--view-img', is_flag=True, default=False, help='show classified image')
def main(accuracy, model, inputdir, outputdir, use_gpu, view_img):
    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')

    # get input data
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    IMAGE_PATHS = []
    for file in glob.glob(inputdir + "*.jpg"):
        print(file.split("\\")[-1])
        IMAGE_PATHS.append(inputdir + file.split("\\")[-1])

    # setup output
    MODEL_DATE = "20200711"
    MODEL_NAME = model
    OUTPUT_DIR = outputdir + MODEL_NAME + "/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CSV_OUTPUT_FILENAME = OUTPUT_DIR + "pred_" + MODEL_NAME + ".csv"
    CSV_OUTPUT_FILE = open(CSV_OUTPUT_FILENAME, 'w', encoding='UTF8', newline='')
    CSV_OUTPUT_WRITER = csv.writer(CSV_OUTPUT_FILE)
    CSV_OUTPUT_WRITER.writerow(["file", "count"])

    PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
    PATH_TO_LABELS = download_labels(LABEL_FILENAME)

    ######################################################################################################################
    # Carregamento do modelo
    ######################################################################################################################

    PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
    PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

    print('Loading model... ', end='')
    start_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    ######################################################################################################################
    # Categorias (classes)
    ######################################################################################################################
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    ######################################################################################################################
    # Detecções
    ######################################################################################################################

    det = None
    rodada = 1

    for image_path in IMAGE_PATHS:
        print('Running inference for {}... '.format(image_path), end='')
        imgname = os.path.splitext(image_path)[0].split("/")[-1].split(".")[0]
        img_classification_file = open(OUTPUT_DIR + imgname + ".txt", "w")

        image_np, width, height = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        det = detections
        print("NUM DETECTIONS", num_detections)

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        detbx = np.empty((0, 4), int)
        detcz = np.empty((0, 4), int)
        detsc = np.empty((0, 4), int)

        clazzes = det['detection_classes']
        count = 0
        for i in range(len(clazzes)):
            score = det['detection_scores'][i]
            if clazzes[i] == 0 and score >= accuracy:
                count = count + 1
                detbx = np.vstack((detbx, det['detection_boxes'][i]))
                detcz = np.append(detcz, det['detection_classes'][i])
                detsc = np.append(detsc, det['detection_scores'][i])
                top = height * det['detection_boxes'][i][0]
                left = width * det['detection_boxes'][i][1]
                bottom = height * det['detection_boxes'][i][2]
                right = width * det['detection_boxes'][i][3]
                print(det['detection_classes'][i], det['detection_scores'][i], left, top, right, bottom,
                      file=img_classification_file)

        CSV_OUTPUT_WRITER.writerow([imgname, count])

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detbx,
            detcz + label_id_offset,
            detsc,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=accuracy,
            agnostic_mode=False)

        plt.figure(figsize=(18, 16))
        plt.imshow(image_np_with_detections)

        if (view_img):
            plt.show()

        plt.savefig(OUTPUT_DIR + imgname + ".jpg")
        img_classification_file.close()
        print("RODADA", rodada, "de", len(IMAGE_PATHS))
        print("MODEL", model)
        rodada = rodada + 1

    print('Done')


if __name__ == '__main__':
    main()
