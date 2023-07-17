import tensorflow as tf


# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from keras.models import load_model
import os.path as osp

import pandas as pd
import os
import keras 
from time import time

from keras_preprocessing.image import img_to_array, array_to_img

if __name__ == "__main__":
    import utils.utils
    from utils.old.fullimagepointcroppingloader import FullImagePointCroppingLoader
else:
    import inferencer.utils.utils
    from inferencer.utils.old.fullimagepointcroppingloader import FullImagePointCroppingLoader   

def convertTime(seconds):
    mins, sec = divmod(seconds, 60)
    hour, mins = divmod(mins, 60)
    if hour > 0:
        return "{:.0f} hour, {:.0f} minutes".format(hour, mins)
    elif mins > 0:
        return "{:.0f} minutes".format(mins)
    else:
        return "{:.0f} seconds".format(sec)

def inference(run_name,
              logs_path,
              input_csv_path,
              output_csv_path,
              image_path_key,
              point_x_key,
              point_y_key,
              label_key,
              cut_divisor):

    IMG_PATH_KEY = image_path_key
    POINT_X_KEY = point_x_key
    POINT_Y_KEY = point_y_key
    LABEL_KEY = label_key

    #RUN_NAME = "groupcode-dense-bayesian2"
    #RUN_NAME = "NWSS-LTM-6854-GroupCode-Bayesian"
    #LOGS = "../logs/" + RUN_NAME + "/"
    #input_fld = LOGS + '/checkpoints-' + RUN_NAME + '/'
    #weight_file = 'weights.best.hdf5'

    #RUN_NAME = "NWSS-LTM-6854-ensemble-data-lifeforms4"
    #LOGS = "/media/mat/NVMESSD/NWSS/logs/" + RUN_NAME + "/"
    #input_fld = LOGS + '/checkpoints-' + RUN_NAME + '/'
    #weight_file = '/media/mat/NVMESSD/NWSS/logs/NWSS-LTM-LIFEFORM4-6854-ensemble-lifeforms4-gbrinit-1/checkpoints-NWSS-LTM-LIFEFORM4-6854-ensemble-lifeforms4-gbrinit-1/weights.best.hdf5'
    #weight_file = '/media/mat/NVMESSD/NWSS/logs/NWSS-LTM-LIFEFORM4-6854-ensemble-lifeforms4-gbrinit-biggerpatch-1/checkpoints-NWSS-LTM-LIFEFORM4-6854-ensemble-lifeforms4-gbrinit-biggerpatch-1/weights.best.hdf5'

    # open the log properties
    LOGS = logs_path + run_name + "/"
    input_fld = LOGS + '/checkpoints-' + run_name + '/'
    weight_file = 'weights.best.hdf5'
    new_mean_image = Image.open(LOGS + "/mean_image.jpg")

    with open(LOGS + "labels.txt") as f:
        labels = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    labels = [x.strip() for x in labels]

    # load the weights
    weight_file_path = osp.join(input_fld, weight_file)

    # load the model
    model = load_model(weight_file_path)

    df = pd.read_csv(input_csv_path)

    X=[]
    y=[]

    for index, row in df.iterrows():
        X.append({"image_path": row[IMG_PATH_KEY], "point_x": row[POINT_X_KEY], "point_y": row[POINT_Y_KEY]})
        y.append(row[LABEL_KEY])


    import random
    def cropping_function(image_dict):
        #print(image_dict)
        patch = utils.load_image_and_crop(image_dict["image_path"], image_dict["point_x"], image_dict["point_y"], 256, 256,cut_divisor=cut_divisor)
        patch = img_to_array(patch)
        #print(image_dict["image_path"])
        #patch -= new_mean_image
        #patch /= 255
        #to_save = keras.preprocessing.image.array_to_img(patch)
        #to_save.save("../data/patches/" +'_255_' +str(random.randint(0, 1000)) + ".jpg")
        return patch

    batch_size = 32
    val_generator = FullImagePointCroppingLoader(X, y,
                                                 batch_size,
                                                 cropping_function)

    tic = time()
    print('Starting inference...')
    predictions = model.predict_generator(val_generator,
                                          steps=(len(y) // batch_size)+1,
                                          verbose=1,
                                          workers=10)

    print('Completed inference of {} points in {}'.format(len(df), convertTime(time()-tic)))
    # pack the predictions and append them to the dataframe
    predicted_labels = []
    predicted_probs = []
    for prediction in predictions:
        predicted = labels[np.argmax(prediction)]
        prediction_prob = np.amax(prediction)

        predicted_labels.append(predicted)
        predicted_probs.append(prediction_prob)

    df["prediction"] = predicted_labels
    df["prediction_prob"] = predicted_probs

    df.to_csv(output_csv_path)
    print('Saved inference results to {}'.format(output_csv_path))
    #pd.DataFrame(results).to_csv("/home/mat/Dev/benthic-automation/experiments/gan_correction/data/slovid.csv")
    #pd.DataFrame(results).to_csv("/home/mat/Dev/benthic-automation/experiments/gan_correction/data/slovidcorrected.csv")


if __name__ == '__main__':
    cut_divisor=8
    inference("LTM_RANDOM_PATCH_6_8_10_12_WATER_AUG-20201026004109",
              "../data/reefcloud-training-runs/",
              #"/home/mat/Desktop/NWSS/Trip6854-SLO-Valid-Points-to-inference.csv",
              "../data/reefcloud-point-exports/LTM_FG_TL_CHECK.csv",
              "../data/reefcloud-inference-results/LTM_FG_TL_CHECK_randpatch_model_div8-preds.csv",
              "image_path",
              "pointx",
              "pointy",
              "code",
              cut_divisor)
    #inference("NWSS-LTM-LIFEFORM4-6854-ensemble-lifeforms4-gbrinit-biggerpatch-1",
    #          "/media/mat/NVMESSD/NWSS/logs/",
    #          #"/home/mat/Desktop/NWSS/Trip6854-SLO-Valid-Points-to-inference.csv",
    #          "/home/mat/Desktop/NWSS/Trip6854-SLO-Valid-Points-to-inference-corrected.csv",
    #          "/home/mat/Dev/benthic-automation/experiments/gan_correction/data/slovidcorrected4.csv",
    #          "LOCAL_PATH",
    #          "POINT_X",
    #          "POINT_Y",
    #          "WA_LIFEFORM4")
