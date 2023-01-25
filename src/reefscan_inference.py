import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from time import time
import joblib

from sklearn.metrics import classification_report

try:
    from keras_preprocessing.image import array_to_img, load_img, img_to_array
except:
    from keras.preprocessing.image import array_to_img, load_img, img_to_array
from keras.models import load_model, Model

import utils
from inference_csv import convertTime
from old.fullimagepointcroppingloader import FullImagePointCroppingLoader


def load_image_and_crop_localfile(image_path, point_x, point_y, crop_width, crop_height, cut_divisor=8):
    img = Image.open(image_path)
    width, height = img.size
    cut_width = int(height/cut_divisor)
    cut_height = int(height/cut_divisor)
    img = utils.cut_patch(img, cut_width, cut_height, point_x, point_y)
    img = img.resize((crop_width, crop_height), Image.NEAREST)    
    return img

# Based on a given key, it copies the column of df_reference
# to df_main if the key exists, otherwise create column with empty strings
def make_columns(keylist, df_main, df_reference):
    df = df_main.copy()
    for key in keylist:
        df[key] = df_reference[key] if key in df_reference.columns else ""
    return df

def read_csv_and_get_relevant_fields(csvpath, localimagedir):
    df_input = pd.read_csv(csvpath)

    # Extract relevant fields from data
    df = df_input[['image_name', 'point_num', 'point_coordinate']]

    optional_fields = ['image_id', 'point_id', 'point_human_classification']
    df = make_columns(optional_fields, df, df_input)

    # Get individual coordinates; append image name to where the local folder is
    df['point_x'] = df.apply(lambda row: int(str(row.point_coordinate).split(',')[0]), axis=1)
    df['point_y'] = df.apply(lambda row: int(str(row.point_coordinate).split(',')[1]), axis=1)
    df['image_path'] = df.apply(lambda row: str(os.path.join(localimagedir, row.image_name)), axis=1)

    # Remove rows with missing fields
    df = df.dropna(axis=0)

    df = df.reset_index()

    return df


# Modified version of `inference` from vector_experiments/src/inference_csv.py
def infer_features(input_data, 
              feature_extractor, 
              image_path_key='image_path',  
              point_x_key='point_x', 
              point_y_key='point_y', 
              label_key='point_human_classification',
              cut_divisor=12,
              feature_out_path='../models/features.csv'):

    IMG_PATH_KEY = image_path_key
    POINT_X_KEY = point_x_key
    POINT_Y_KEY = point_y_key
    LABEL_KEY = label_key

    feature_layer_name = 'global_average_pooling2d_1' #'avg_pool'
    model = load_model(feature_extractor)
    model = Model(inputs=model.inputs,
                  outputs=model.get_layer(feature_layer_name).output)
    df = input_data.copy() 

    print(df)

    X = []
    y = []


    if LABEL_KEY == 'unlabelled':
        for index, row in df.iterrows():
            X.append({"image_path": row[IMG_PATH_KEY], "point_x": row[POINT_X_KEY], "point_y": row[POINT_Y_KEY]})
            y.append(LABEL_KEY)
    else: 
        for index, row in df.iterrows():
            X.append({"image_path": row[IMG_PATH_KEY], "point_x": row[POINT_X_KEY], "point_y": row[POINT_Y_KEY]})
            y.append(row[LABEL_KEY])

    import random
    def cropping_function(image_dict):
        patch = load_image_and_crop_localfile(image_dict["image_path"], image_dict["point_x"], image_dict["point_y"], 256, 256,cut_divisor=cut_divisor)
        patch = img_to_array(patch)
        return patch

    batch_size = 32
    val_generator = FullImagePointCroppingLoader(X, y, batch_size, cropping_function)


    tic = time()
    print('Starting inference...')
    predictions = model.predict_generator(val_generator,
                                          steps=(len(y) // batch_size)+1,
                                          verbose=1,
                                          workers=10)

    print('Completed (feature extraction) inference of {} points in {}'.format(len(df), convertTime(time()-tic)))

    # create columns for feature vectors
    print(np.shape(predictions), np.shape(predictions[0]))
    
    cols = []
    for i in range(len(predictions[0])):
        cols.append('feature_vector_{}'.format(i))
    df_vects = pd.DataFrame(predictions, columns=cols)

    df = pd.concat([df, df_vects], axis=1)
    df.to_csv(feature_out_path, index=False)
    print('Saved inference results to {}'.format(feature_out_path))

    # # pack the predictions and append them to the dataframe
    # predicted_labels = []
    # predicted_probs = []
    # for prediction in predictions:
    #     predicted = labels[np.argmax(prediction)]
    #     prediction_prob = np.amax(prediction)

    #     predicted_labels.append(predicted)
    #     predicted_probs.append(prediction_prob)

    # df["prediction"] = predicted_labels
    # df["prediction_prob"] = predicted_probs

    df.to_csv(feature_out_path)
    print('Saved inference results to {}'.format(feature_out_path))

    return df


def infer_class(features_data, model, scaler, encoder, group_labels_csv_file, output_results_file, output_coverage_file):
    def get_XY(df, n_features=128, label_col='encoded_label'):
        feat_cols = []
        for i in range(n_features):
            feat_cols.append('feature_vector_{}'.format(i))

        X = df[feat_cols].to_numpy()
        y = df[label_col].to_numpy()

        return X, y

    X, y = get_XY(features_data, label_col='point_human_classification')

    X_transformed = scaler.fit_transform(X, y)

    print("ENCODER CLASSES:\n\n")
    print(list(encoder.classes_))

    yhat = model.predict(X_transformed)


    true_labels = y
    pred_labels = encoder.inverse_transform(yhat)


    print(classification_report(true_labels, pred_labels))

    df_results = features_data[['image_path', 'image_id', 'point_num', 'point_id', 'point_coordinate']]
    df_results['true_class'] = true_labels
    df_results['pred_class'] = pred_labels

    df_labels = pd.read_csv(group_labels_csv_file)
    desc_mapping = dict(zip(df_labels["code"], df_labels["description"]))
    grp_mapping =  dict(zip(df_labels["code"], df_labels["group_code"]))

    df_results['true_group'] = df_results['true_class'].map(grp_mapping)
    df_results['pred_group'] = df_results['pred_class'].map(grp_mapping)

    print(df_results)
    df_results.to_csv(output_results_file)

    df_coverage = pd.DataFrame()
    df_coverage['true_counts'] = df_results.groupby('true_group').size()
    df_coverage['true_percentage'] = [element / df_coverage['true_counts'].sum() for element in df_coverage['true_counts']]
    df_coverage['pred_counts'] = df_results.groupby('pred_group').size()
    df_coverage['pred_percentage'] = [element / df_coverage['pred_counts'].sum() for element in df_coverage['pred_counts']]
    df_coverage = df_coverage.fillna(0)
    df_coverage.index.names = ['group']
    df_coverage.loc['total'] = df_coverage.iloc[:].sum()

    print(df_coverage)
    df_coverage.to_csv(output_coverage_file)


def inference(feature_extractor='../models/ft_ext/weights.best.hdf5', 
              classifier='../models/classifier/reefscan.sav',
              group_labels_csv_file='../models/reefscan_group_labels.csv', 
              points_csv_file='../data/reefscan_points.csv',
              local_image_dir='../data/input_images', 
              image_path_key='image_path', 
              point_x_key='pointx', 
              point_y_key='pointy', 
              label_key='point_human_classification',
              cut_divisor=12,
              intermediate_feature_outputs_path='../models/features.csv',
              output_results_file='../results/results.csv',
              output_coverage_file='../results/coverage-summary.csv',
              use_cache='False'):

    using_cache = False if use_cache.lower() in ['False','false'] else True

    df_input = read_csv_and_get_relevant_fields(points_csv_file, local_image_dir)

    # To save time and avoid redoing the feature extraction, the program will first check
    # whether there exists a csv file with the extracted features  
    if using_cache and os.path.exists(intermediate_feature_outputs_path):
        print("Using saved features from previous feature extraction inference")
        df_features = pd.read_csv(intermediate_feature_outputs_path)
    else:
        print("Performing feature extraction...")
        df_features = infer_features(df_input, feature_extractor, feature_out_path=intermediate_feature_outputs_path)

    model, scaler, encoder = joblib.load(classifier)
    infer_class(df_features, model, scaler, encoder, group_labels_csv_file, output_results_file, output_coverage_file)

def main(**kwargs):
    print('Starting inference')

    for key, value in kwargs.items():
        print(f'Arg: {key}={value}')
    
    inference(**kwargs)

if __name__ == '__main__':
    sysargs = [arg.replace("--","") for arg in sys.argv[1:]]
    main(**dict(arg.split('=') if '=' in arg else [arg, True] for arg in sysargs))

