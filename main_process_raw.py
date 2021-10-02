# Preâmbulo
import csv
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

######################################################################################################################
# VARIÁVEIS
######################################################################################################################
tf.config.set_visible_devices([], 'GPU')
matplotlib.use("Qt5Agg")
ACCURACY = 0.2
MODEL_DATE = "20200711"
MODEL_NAME = "efficientdet_d4_coco17_tpu-32"
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
OUTPUT_DIR = "C:/Imagens/Output/RAW/" + MODEL_NAME + "/"
CSV_OUTPUT_FILENAME = OUTPUT_DIR + "pred_" + MODEL_NAME + ".csv"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CSV_OUTPUT_FILE = open(CSV_OUTPUT_FILENAME, 'w', encoding='UTF8', newline='')
CSV_OUTPUT_WRITER = csv.writer(CSV_OUTPUT_FILE)
CSV_OUTPUT_WRITER.writerow(["file", "count"])

# IMAGE_PATHS = ["C:/Imagens/RAW/34061.jpg", "C:/Imagens/RAW/32582.jpg"]
INPUT_DIR = "C:/Imagens/RAW/"
IMAGE_PATHS = []
for file in glob.glob(INPUT_DIR + "*.jpg"):
    print(file.split("\\")[-1])
    IMAGE_PATHS.append(INPUT_DIR + file.split("\\")[-1])

# IMAGE_PATHS = ["C:/Imagens/RAW/31930.jpg", "C:/Imagens/RAW/34061.jpg", "C:/Imagens/RAW/32582.jpg"]


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


PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
print("MODELO", PATH_TO_MODEL_DIR)


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


PATH_TO_LABELS = download_labels(LABEL_FILENAME)
print("LABELS", PATH_TO_LABELS)

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


@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


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
rodada = 0


def load_image_into_numpy_array(path):
    img = Image.open(path)
    return np.array(img), img.width, img.height


for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')
    imgname = os.path.splitext(image_path)[0].split("/")[-1].split(".")[0]
    img_classification_file = open(OUTPUT_DIR + imgname + ".txt", "w")

    image_np, width, height = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

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
        if clazzes[i] == 0 and score >= ACCURACY:
            count = count + 1
            detbx = np.vstack((detbx, det['detection_boxes'][i]))
            detcz = np.append(detcz, det['detection_classes'][i])
            detsc = np.append(detsc, det['detection_scores'][i])
            top = height * det['detection_boxes'][i][0]
            left = width * det['detection_boxes'][i][1]
            bottom = height * det['detection_boxes'][i][2]
            right = width * det['detection_boxes'][i][3]
            print(det['detection_classes'][i], det['detection_scores'][i], left, top, right, bottom)
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
        min_score_thresh=ACCURACY,
        agnostic_mode=False)

    print("---------------------------------------------------------------")
    print(detcz, detbx, detsc)
    plt.figure(figsize=(18, 16))
    plt.imshow(image_np_with_detections)
    # plt.show()
    plt.savefig(OUTPUT_DIR + imgname + ".jpg")
    img_classification_file.close()
    print("RODADA", rodada, "de", len(IMAGE_PATHS))
    rodada = rodada + 1
    print('Done')
# plt.show()

# sphinx_gallery_thumbnail_number = 2
