# Preâmbulo
import click
import csv
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
import tensorflow as tf
import time

from PIL import Image
from postprocess import PostProcessing
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Constants and setup info
matplotlib.use("Qt5Agg")
det_box_id = 0


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


def predict(image_np, width, height, detection_model, accuracy):
    global det_box_id

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

    # All outputs are batches tensors.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    outdetections = []

    detbx = np.empty((0, 4), int)
    detcz = np.empty((0, 4), int)
    detsc = np.empty((0, 4), int)

    clazzes = detections['detection_classes']
    for i in range(len(clazzes)):
        score = detections['detection_scores'][i]
        if clazzes[i] == 0 and score >= accuracy:
            detbx = np.vstack((detbx, detections['detection_boxes'][i]))
            detcz = np.append(detcz, detections['detection_classes'][i])
            detsc = np.append(detsc, detections['detection_scores'][i])

            top = height * detections['detection_boxes'][i][0]
            left = width * detections['detection_boxes'][i][1]
            bottom = height * detections['detection_boxes'][i][2]
            right = width * detections['detection_boxes'][i][3]
            det_box_id = det_box_id + 1

            outdetections.append({
                "id": det_box_id,
                "clazz": detections['detection_classes'][i],
                "conf": detections['detection_scores'][i],
                "xmin": left,
                "xmax": right,
                "ymin": top,
                "ymax": bottom
            })
            print(detections['detection_classes'][i], detections['detection_scores'][i], left, top, right, bottom)

    return outdetections, detbx, detcz, detsc


@click.command()
@click.option('--accuracy', type=float, default=0.2, help='accuracy threshold')
@click.option('--model', type=str, default='efficientdet_d4_coco17_tpu-32',
              help='model name (see https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md')
@click.option('--inputdir', type=str, default='C:/Imagens/RAW/', help='input folder')
@click.option('--outputdir', type=str, default='C:/Imagens/Output/TILING/', help='output folder')
@click.option('--numtile', type=int, default=4, help="numero de tiles")
@click.option('--use-gpu', is_flag=True, default=False, help='should we use gpu')
@click.option('--view-img', is_flag=True, default=False, help='show classified image')
def main(accuracy, model, inputdir, outputdir, numtile, use_gpu, view_img):
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
    CSV_OUTPUT_WRITER.writerow(["file", "count", "cnt_before"])

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
    # Putting additional category name as person so that we can show the person in other tiles with different colors
    # the idea is that persons in a given tile with receive a diff category
    # |--------|--------|
    # |   1    |    2   |
    # -------------------
    # |   3    |   4    #
    # -------------------
    category_index[2]["name"] = "person"
    category_index[3]["name"] = "person"
    category_index[4]["name"] = "person"

    ######################################################################################################################
    # Detecções
    ######################################################################################################################

    det = None
    rodada = 1

    for image_path in IMAGE_PATHS:
        print('Running inference for {}... '.format(image_path), end='')
        imgname = os.path.splitext(image_path)[0].split("/")[-1].split(".")[0]
        img_classification_file = open(OUTPUT_DIR + imgname + ".txt", "w")

        img = cv2.imread(image_path)
        postimg_with_boxes = img.copy()
        postimg_with_merge_boxes = img.copy()

        height, width = img.shape[:2]
        ym, xm = int(height / 2), int(width / 2)

        # Post Processor object
        tam_slice_width = xm
        tam_slice_height = ym
        post = PostProcessing(tam_total_height=height, tam_total_width=width,
                              tam_slice_height=tam_slice_height, tam_slice_width=tam_slice_width)

        i0 = img[:ym, :xm, :]
        i1 = img[:ym, xm:, :]
        i2 = img[ym:, :xm, :]
        i3 = img[ym:, xm:, :]

        outpieces = []

        imgpieces = [i0, i1, i2, i3]
        label_id_offset = 1

        for ix in imgpieces:
            outdetections, detbx, detcz, detsc = predict(ix, xm, ym, detection_model, accuracy)
            outpieces.append(outdetections)
            if view_img:
                # A parte abaixo mostra as detecções em cada box
                image_np_with_detections = ix.copy()

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
                plt.show()

        # Expande as bbs detectadas
        bboriginal = post.expand_bounding_box(outpieces[0], outpieces[1], outpieces[2], outpieces[3])
        cnt_before = len(bboriginal)
        print("BBORIGINAL", bboriginal)
        print("TAMANHO BB ORIG", cnt_before)

        detbx = np.empty((0, 4), int)
        detcz = np.empty((0, 4), int)
        detsc = np.empty((0, 4), int)

        for bx in bboriginal:
            bx_limits = np.array([bx["ymin"], bx["xmin"], bx["ymax"], bx["xmax"]], dtype="float32")
            detbx = np.vstack((detbx, bx_limits))
            detcz = np.append(detcz, bx["clz"])
            detsc = np.append(detsc, bx["conf"])

        viz_utils.visualize_boxes_and_labels_on_image_array(
            postimg_with_boxes,
            detbx,
            detcz + 1,
            detsc,
            category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=200,
            min_score_thresh=accuracy,
            agnostic_mode=False)
        plt.figure(figsize=(18, 16))
        plt.imshow(postimg_with_boxes)

        if view_img:
            plt.show()
        else:
            plt.savefig(OUTPUT_DIR + imgname + "_tiling_before_" + MODEL_NAME + ".jpg")

        outfixed = post.fix_bounding_box(outpieces[0], outpieces[1], outpieces[2], outpieces[3])
        cnt_merge = len(outfixed)
        CSV_OUTPUT_WRITER.writerow([imgname, cnt_merge, cnt_before])

        detbx = np.empty((0, 4), int)
        detcz = np.empty((0, 4), int)
        detsc = np.empty((0, 4), int)

        print("DEPOIS DE USAR MERGE", cnt_merge)
        for bx in outfixed:
            bx_limits = np.array([bx["ymin"], bx["xmin"], bx["ymax"], bx["xmax"]], dtype="float32")
            detbx = np.vstack((detbx, bx_limits))
            detcz = np.append(detcz, 0)
            detsc = np.append(detsc, bx["conf"])
            print(0, bx["conf"], bx["xmin"], bx["ymin"], bx["xmax"], bx["ymax"], file=img_classification_file)

        viz_utils.visualize_boxes_and_labels_on_image_array(
            postimg_with_merge_boxes,
            detbx,
            detcz + 1,
            detsc,
            category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=200,
            min_score_thresh=accuracy,
            agnostic_mode=False)

        plt.figure(figsize=(18, 16))
        plt.imshow(postimg_with_merge_boxes)
        if view_img:
            plt.show()
        else:
            plt.savefig(OUTPUT_DIR + imgname + "_tiling_merge_" + MODEL_NAME + ".jpg")

        img_classification_file.close()
        print("RODADA", rodada, "de", len(IMAGE_PATHS))
        print("MODEL", model)
        rodada = rodada + 1

    print('Done')
    CSV_OUTPUT_FILE.close()


if __name__ == '__main__':
    main()
