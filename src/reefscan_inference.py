import os
import sys

import numpy as np
import pandas as pd
import PIL
from PIL import Image
from time import time
import joblib
import tensorflow as tf

from sklearn.metrics import classification_report

try:
    from keras_preprocessing.image import array_to_img, load_img, img_to_array
except:
    from keras.preprocessing.image import array_to_img, load_img, img_to_array
from keras.models import load_model, Model

import utils
from inference_csv import convertTime
from old.fullimagepointcroppingloader import FullImagePointCroppingLoader

DEFAULT_COORDS=['1328, 760',
'3984, 760',
'2656, 1520',
'1328, 2280',
'3984, 2280']

GLOBAL_REMOVE_LIST = []

def get_index_of_extension(name):
    return name.find(f".{name.split('.')[-1]}")

def isImageFile(name):
    ext = name[get_index_of_extension(name)+1:]
    valid_ext = ['jpg','jpeg','png','bmp','svg','webp','ico','tiff','avif']
    return ext.lower() in valid_ext

def load_image_and_crop_localfile(image_path, point_x, point_y, crop_width, crop_height, cut_divisor=8):
    try:
        img = Image.open(image_path)
        width, height = img.size
        cut_width = int(height/cut_divisor)
        cut_height = int(height/cut_divisor)
        img = utils.cut_patch(img, cut_width, cut_height, point_x, point_y)
        img = img.resize((crop_width, crop_height), Image.NEAREST)
        return img
    except PIL.UnidentifiedImageError:    
        print(f'Error loading image {image_path}: image is invalid or corrupt')
        global GLOBAL_REMOVE_LIST
        GLOBAL_REMOVE_LIST.append(image_path)
        return Image.new(mode='RGB', size=(crop_width, crop_height))


# Based on a given key, it copies the column of df_reference
# to df_main if the key exists, otherwise create column with empty strings
def make_columns(keylist, df_main, df_reference):
    df = df_main.copy()
    for key in keylist:
        df[key] = df_reference[key] if key in df_reference.columns else ""
    return df


# Creates a points dataframe that terates through the local test directory and
# adds arbitrary coordinate values and point numbers similar to the typical exported reefscan points csv
def create_points_df(localimagedir):
    df = pd.DataFrame()
    files_list = os.listdir(localimagedir)

    df['image_path'] = [os.path.join(localimagedir, f) for f in files_list for i in range(5) ]
    df['point_coordinate'] = [default_coord for i in range(len(files_list)) for default_coord in DEFAULT_COORDS]
    df['point_num'] = [j+1 for i in range(len(files_list)) for j in range(5)]


    # Get individual coordinates; append image name to where the local folder is
    df['point_x'] = df.apply(lambda row: int(str(row.point_coordinate).split(',')[0]), axis=1)
    df['point_y'] = df.apply(lambda row: int(str(row.point_coordinate).split(',')[1]), axis=1)

    # Remove rows that are non-image files
    df = df[[isImageFile(str(img)) for img in df.image_path]]

    # Remove rows with missing fields
    df = df.dropna(axis=0)

    df = df.reset_index()

    optional_fields = ['image_id', 'point_id', 'point_human_classification']
    df = make_columns(optional_fields, df, df)

    print(df)

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
    # df = df.dropna(axis=0)
    df = df.fillna('None')

    df = df.reset_index()


    return df


def split_df(df, batch_size):
    array_of_dfs = []
    array_size = int(len(df) / batch_size) + 1
    for i in range(array_size):
        start_idx = i * batch_size
        end_idx = np.minimum(len(df), (i+1) * batch_size)
        array_of_dfs.append(df.iloc[start_idx : end_idx])
    return array_of_dfs

def merge_df(array_of_dfs):
    return pd.concat(array_of_dfs, ignore_index=False)

def load_saved_state(saved_state_csv):
    finished_df = pd.read_csv(saved_state_csv)
    finished_df = finished_df.fillna('None')
    print(f'\n###\n#\n# Last saved at dataframe index {finished_df.index[-1]} \n#\n###\n')
    return finished_df, finished_df.index[-1]


# Modified version of `inference` from vector_experiments/src/inference_csv.py
def infer_features(input_data, 
              feature_extractor, 
              image_path_key='image_path',  
              point_x_key='point_x', 
              point_y_key='point_y', 
              label_key='point_human_classification',
              cut_divisor=12,
              feature_out_path='../models/features.csv',
              temp_feature_out_path='../models/temp_features.csv'):


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


    def cropping_function(image_dict):
        patch = load_image_and_crop_localfile(image_dict["image_path"], image_dict["point_x"], image_dict["point_y"], 256, 256,cut_divisor=cut_divisor)
        patch = img_to_array(patch)            
        return patch



    saved_state_batch_size = 512
    array_of_dfs = split_df(df, saved_state_batch_size)

    loaded_df = pd.DataFrame()

    # Check if dataframe was previously cached and saved in a temp file
    if os.path.exists(temp_feature_out_path):
        print(f'\n###\n#\n# FOUND TEMPORARY SAVED STATE FILE\n#\n###\n')
        loaded_df, saved_state_cursor = load_saved_state(temp_feature_out_path)
        # Check if cache is actually matching the image data path
        if loaded_df.at[0, 'image_path'] == df.at[0, 'image_path']:
            array_of_dfs = split_df(df.iloc[saved_state_cursor+1 :], saved_state_batch_size)

    tic = time()
    print('Starting inference...')

    for df_idx, part_df in enumerate(array_of_dfs):
        print(f'Processing this particular DF: \n {part_df} \n :::::')

        X = []
        y = []
        if LABEL_KEY == 'unlabelled':
            for index, row in part_df.iterrows():
                X.append({"image_path": row[IMG_PATH_KEY], "point_x": row[POINT_X_KEY], "point_y": row[POINT_Y_KEY]})
                y.append(LABEL_KEY)
        else: 
            for index, row in part_df.iterrows():
                X.append({"image_path": row[IMG_PATH_KEY], "point_x": row[POINT_X_KEY], "point_y": row[POINT_Y_KEY]})
                y.append(row[LABEL_KEY])


        batch_size = 32

        val_generator = FullImagePointCroppingLoader(X, y, batch_size, cropping_function)



        if len(y) % batch_size == 0:
            steps = len(y) // batch_size
        else:
            steps = (len(y) // batch_size) + 1

        predictions = model.predict_generator(val_generator,
                                            steps=steps,
                                            verbose=1,
                                            workers=10)

        print('Completed (feature extraction) inference of {} points in {}'.format(len(part_df), convertTime(time()-tic)))
    

        # create columns for feature vectors
        print(np.shape(predictions), np.shape(predictions[0]))
        
        # cols = []
        # for i in range(len(predictions[0])):
        #     cols.append('feature_vector_{}'.format(i))
        # df_vects = pd.DataFrame(predictions, columns=cols)

        # part_df = pd.concat([part_df, df_vects], axis=1)


        for i in range(len(predictions[0])):
            part_df[f'feature_vector_{i}'] = predictions[:, i]

        header_required = not os.path.exists(temp_feature_out_path)

        part_df = part_df[~part_df['image_path'].isin(GLOBAL_REMOVE_LIST)]

        part_df.to_csv(temp_feature_out_path, mode='a', index=False, header=header_required)
        print('Saved inference results to {}'.format(temp_feature_out_path))

        print(f'DF After processing: \n {part_df} \n :::::')


        array_of_dfs[df_idx] = part_df

    df = merge_df(array_of_dfs)
    if os.path.exists(feature_out_path):
        os.remove(feature_out_path)
    os.rename(temp_feature_out_path, feature_out_path)

    if not loaded_df.empty:
        df = pd.concat([loaded_df, df], axis=0)

    print(df)

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

    df_results['true_desc'] = df_results['true_class'].map(desc_mapping)
    df_results['pred_desc'] = df_results['pred_class'].map(desc_mapping)

    df_results['true_group'] = df_results['true_class'].map(grp_mapping)
    df_results['pred_group'] = df_results['pred_class'].map(grp_mapping)


    print(df_results)
    df_results.to_csv(output_results_file)


    df_coverage = pd.DataFrame()
    df_results.drop(df_results.loc[df_results['true_group']=='IN'].index, inplace=True)
    df_results.drop(df_results.loc[df_results['pred_group']=='IN'].index, inplace=True)

    df_coverage['true_counts'] = df_results.groupby('true_group').size()
    df_coverage['true_percentage'] = [element / df_coverage['true_counts'].sum() for element in df_coverage['true_counts']]
    df_coverage['pred_counts'] = df_results.groupby('pred_group').size()
    df_coverage['pred_percentage'] = [element / df_coverage['pred_counts'].sum() for element in df_coverage['pred_counts']]
    df_coverage = df_coverage.fillna(0)
    df_coverage.index.names = ['group']
    df_coverage.loc['total'] = df_coverage.iloc[:].sum()

    print(df_coverage)
    df_coverage.to_csv(output_coverage_file)

def str2bool(mystr):
    return False if mystr.lower() in ['0','false'] else True

def inference(feature_extractor='../models/ft_ext/weights.best.hdf5', 
              classifier='../models/classifier/reefscan.sav',
              group_labels_csv_file='../models/reefscan_group_labels.csv', 
              points_csv_file='',
              local_image_dir='../data/input_images', 
              image_path_key='image_path', 
              point_x_key='pointx', 
              point_y_key='pointy', 
              label_key='point_human_classification',
              cut_divisor=12,
              intermediate_feature_outputs_path='../models/features.csv',
              output_results_file='../results/results.csv',
              output_coverage_file='../results/coverage-summary.csv',
              saved_state_file='../data/saved_state.csv',
              use_cache='False',
              use_gpu='False'):

    # Resolve string args to bools
    using_cache = str2bool(use_cache)
    using_gpu = str2bool(use_gpu)

    # GPU Setup if using_gpu
    if using_gpu:
        print('Confirm TF is using gpu')
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print(gpu_devices)
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    if points_csv_file:
        df_input = read_csv_and_get_relevant_fields(points_csv_file, local_image_dir)
    else:
        df_input = create_points_df(local_image_dir)


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
    
    overall_start = time()
    inference(**kwargs)
    overall_end = time()

    print(f'Total elapsed time is {overall_end - overall_start} seconds')

if __name__ == '__main__':
    sysargs = [arg.replace("--","") for arg in sys.argv[1:]]
    main(**dict(arg.split('=') if '=' in arg else [arg, 'True'] for arg in sysargs))

