

import time
import glob2
import sys
import numpy
from PIL import Image
import numpy as np
import os
import io

try:
    from keras_preprocessing.image import array_to_img, load_img, img_to_array
except:
    from keras.preprocessing.image import array_to_img, load_img, img_to_array

from keras.preprocessing.image import DirectoryIterator
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import Callback, TensorBoard
from keras.utils import GeneratorEnqueuer
from keras.utils import OrderedEnqueuer
from keras.utils import Progbar
from keras.utils import Sequence
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd

import boto3

import warnings
from time import sleep


class MultilabelMetrics(Callback):
    def __init__(self, validation_data, steps, target_names):
        self.validation_data = validation_data
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.steps = steps
        self.target_names = target_names

    def on_epoch_end(self, epoch, logs={}):

        self.model.predict_generator = custom_predict_generator

        y_pred, y_true = self.model.predict_generator(self.model, generator=self.validation_data, steps=self.steps)

        #print(y_pred[0])
        #print(y_true[0])

        ranking_precision = label_ranking_average_precision_score(y_true, y_pred)
        ranking_loss = label_ranking_loss(y_true, y_pred)

        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
        precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        print("Ranking Average Precision: %s - Ranking Loss: %s" % (str(ranking_precision), str(ranking_loss)))
        print("Accuracy: %s - Precision: %s - Recall: %s - F1: %s" % (str(accuracy), str(precision), str(recall), str(f1)))

        fd = open('metriclogs.csv', 'a')
        fd.write("%s, %s, %s, %s, %s, %s \n" % (str(ranking_precision), str(ranking_loss), str(accuracy), str(precision), str(recall), str(f1)))
        fd.close()

        #print(classification_report(y_true, y_pred, target_names=self.target_names))

        return

class Metrics(Callback):
    def __init__(self, validation_data, steps, target_names):
        self.validation_data = validation_data
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.steps = steps
        self.target_names = target_names

    def on_epoch_end(self, epoch, logs={}):

        self.model.predict_generator = custom_predict_generator

        predictions, y_true = self.model.predict_generator(self.model, generator=self.validation_data, steps=self.steps)

        new_y_pred = []
        new_y_true = []
        group_yes_no = {}
        group_scores = {}

        for group in self.target_names:
            group_yes_no[group.strip()] = []
            group_scores[group.strip()] = []

        for index, item in enumerate(predictions):
            #print(np.argmax(item), np.amax(item), np.argmax(y_true[index]))
            original = np.argmax(y_true[index])
            prediction = np.argmax(item)
            prediction_score = np.amax(item)

            new_y_pred.append(prediction)
            new_y_true.append(original)

            original = self.target_names[original]
            prediction = self.target_names[prediction]

            if original == prediction:
                group_yes_no[original].append(1)

            else:
                group_yes_no[original].append(0)

            group_scores[original].append(float(prediction_score))

        print(classification_report(new_y_true, new_y_pred, target_names=self.target_names))

        ps = []
        rs = []
        ts = []

        for group in self.target_names:


            try:
                print("ROC " + group + ": " + str(roc_auc_score(group_yes_no[group], group_scores[group])))
            except ValueError:
                pass

            if len(group_yes_no[group]) == 0:
                group_yes_no[group].append(0)
                group_scores[group].append(0.0)

            precision, recall, thresholds = precision_recall_curve(np.array(group_yes_no[group]),
                                                                   np.array(group_scores[group]))

            prec = precision
            p_temp = precision[0]
            n = len(prec)
            for i in range(n):
                if prec[i] < p_temp:
                    prec[i] = p_temp
            else:
                p_temp = prec[i]

            ps.append(prec)
            rs.append(recall)
            ts.append(thresholds)

        self.plot_precision_recalls(ps, rs, ts, self.target_names, epoch)

        return

    def plot_precision_recalls(self, ps, rs, ts, labels, epoch):

        NUM_COLORS = len(labels)

        cmap = plt.get_cmap('spectral')
        N = len(labels)

        for index, item in enumerate(ps):
            '''
            x = np.array(rs[index])
            y = np.array(item)

            if x.size >= 2 and y.size > 2:
                x_smooth = np.linspace(x.min(), x.max(), 1000)
                ius = InterpolatedUnivariateSpline(x, y)
                y_smooth = ius(x_smooth)

                plt.plot(x_smooth, y_smooth, label=labels[index], c=cmap(float(index) / N))
            '''

            plt.plot(rs[index], item, label=labels[index], c=cmap(float(index) / N))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        plt.title("Precision Recall All Classes")
        plt.legend(loc="lower left")
        plt.savefig(str(epoch) + "p-recall.png")
        plt.clf()
        plt.cla()
        plt.close()

def custom_predict_generator(self,
                      generator,
                      steps,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0,
                      **kwargs):
    """Generates predictions for the input samples from a data generator.

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: Generator yielding batches of input samples
                or an instance of Sequence (keras.utils.Sequence)
                object in order to avoid duplicate data
                when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_queue_size: Maximum size for the generator queue.
        workers: Maximum number of processes to spin up
            when using process based threading
        use_multiprocessing: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.
        **kwargs: support for legacy arguments.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case the generator yields
            data in an invalid format.
    """
    all_ys = []

    # Legacy support
    if 'max_q_size' in kwargs:
        max_queue_size = kwargs.pop('max_q_size')
        logging.warning('The argument `max_q_size` has been renamed '
                        '`max_queue_size`. Update your method calls accordingly.')
    if 'pickle_safe' in kwargs:
        use_multiprocessing = kwargs.pop('pickle_safe')
        logging.warning('The argument `pickle_safe` has been renamed '
                        '`use_multiprocessing`. '
                        'Update your method calls accordingly.')

    self._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    all_indexes = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        logging.warning(
            logging.warning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
    enqueuer = None

    try:
        if is_sequence:
            enqueuer = OrderedEnqueuer(
                generator, use_multiprocessing=use_multiprocessing)
        else:
            enqueuer = GeneratorEnqueuer(
                generator,
                use_multiprocessing=use_multiprocessing,
                wait_time=wait_time)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                #print("gen output length = ", len(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                elif len(generator_output) == 3:
                    x, y, indexes = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' + str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x, y = generator_output
            #indexes = [indexes]
            outs = self.predict_on_batch(x)
            if not isinstance(outs, list):
                outs = [outs]
                y = [y]

            if not all_outs:
                for out in outs:
                    all_outs.append([])
                    all_ys.append([])
                    all_indexes.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)

            for i, yi in enumerate(y):
                all_ys[i].append(yi)

          #  for i, ii in enumerate(indexes):
           #     all_indexes[i].append(ii)

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            return all_outs[0][0], all_ys[0][0], all_indexes[0][0]
        else:
            return np.concatenate(all_outs[0]), np.concatenate(all_ys[0])#, np.concatenate(all_indexes[0])
    if steps_done == 1:
        return [out for out in all_outs], [out for out in all_ys], [out for out in all_indexes]
    else:
        return [np.concatenate(out) for out in all_outs], [np.concatenate(out) for out in all_ys], [np.concatenate(out) for out in all_indexes]


def calculate_mean_image(directory, imgwidth=256, imgheight=256):

    print("Calculating mean image")

    count = 0
    beginTime = time.time()

    # Access all PNG files in directory
    imlist = glob2.glob(directory + "/**/*.jpg")
    #imlist = [filename for filename in allfiles if filename[-4:] in [".jpg"]]

    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(imlist[0]).size
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = np.array(Image.open(im), dtype=np.float)
        arr = arr + imarr / N

        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - beginTime
            print("Processed {} images in {:.2f} seconds. "
                  "{:.2f} images/second.".format(N, elapsed,
                                                 N / elapsed))

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)
    print (arr.shape)
    return arr


def get_rect_dimensions_relative(width, height, patchwidth, patchheight, pointx, pointy):
    return [int((width*pointx)-(patchwidth/2)), int((height*pointy)-(patchheight/2)),
            int((width*pointx)+(patchwidth/2)), int((height*pointy)+(patchheight/2))]

def get_rect_dimensions(width, height, patchwidth, patchheight, pointx, pointy):
    return [int(int(pointx)-(patchwidth/2)), int(int(pointy)-(patchheight/2)),
            int(int(pointx)+(patchwidth/2)), int(int(pointy)+(patchheight/2))]

def get_rect_dimensions_pixels(width, height, patchwidth, patchheight, pointx, pointy):
    return [int((pointx)-(patchwidth/2)), int((pointy)-(patchheight/2)),
            int((pointx)+(patchwidth/2)), int((pointy)+(patchheight/2))]


def cut_patch(image, patch_width, patch_height, x, y):

    width, height = image.size
    dimensions = get_rect_dimensions_pixels(width, height, patch_width, patch_height, x, y)
    new_image = image.crop(dimensions)
    return new_image

#import pyvips
#def load_image_and_crop_n(image_path, point_x, point_y, crop_width, crop_height):
#    #start = time.time()
#
#    img = pyvips.Image.new_from_file(image_path, access='sequential')
#
#    cut_width = int(img.height /24)  # 16)
#    cut_height = int(img.height /24)  # 16)
#
#    dimensions = get_rect_dimensions(img.width, img.height, cut_width, cut_height, point_x, point_y)
#    roi = img.crop(dimensions[0], dimensions[1], dimensions[2]-dimensions[0], dimensions[3]-dimensions[1])
#    #print(dimensions[0], dimensions[1], dimensions[2]-dimensions[0], dimensions[3]-dimensions[1])
#    roi = roi.thumbnail_image(1000, height=crop_height)
#    #roi = pyvips.Image.thumbnail_buffer(roi, 1000, height=crop_height)
#    mem_img = roi.write_to_memory()
#
#    # Make a numpy array from that buffer object
#    nparr = np.ndarray(buffer=mem_img, dtype=np.uint8,
#                       shape=[roi.height, roi.width, roi.bands])
#
#    #print("image took", time.time() - start)
#
#    return nparr
#
# converts image path from csv into a file stream to load images directly from the s3 bucket
#TODO probably best to pass the s3, and bucket variables from the main script so we don't regenerate them for every image
def load_s3_im_from_bucket(image_path, boto3_session, bucket_name):
    # strips /home/ubuntu/images from path to give s3 path
    # image_bucket_key = os.path.join(*(image_path.split(os.path.sep)[4:]))
    image_bucket_key = image_path.replace('/home/ubuntu/images/', '')
    try:
        s3 = boto3_session.resource('s3', region_name='ap-southeast-2')
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(image_bucket_key)
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)
        return file_stream
    except:
        error = 'Error fetching image bucket key ' + image_bucket_key + ' from bucket ' + bucket_name + ' - '  + str(sys.exc_info())
        raise Exception(error)


def load_s3_im(pth, boto3_session, images_bucket_name='reefcloud-image-uploads'):
    # strips /home/ubuntu/images from path to give s3 path
    s3_image_pth = pth#os.path.join(*(pth.split(os.path.sep)[4:]))
    try:
        s3 = boto3_session.resource('s3', region_name='ap-southeast-2')
        #bucket = s3.Bucket('reefcloud-images-20200129033411415400000005')
        bucket = s3.Bucket(images_bucket_name)
        obj = bucket.Object(s3_image_pth)
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)
        return file_stream
    except:
        error = 'Error fetching image ' + s3_image_pth + ' from bucket reefcloud-image-uploads - '  + str(sys.exc_info())
        raise Exception(error)

# local image loading
def load_image_and_crop_localfile(image_path, point_x, point_y, crop_width, crop_height, cut_divisor=8):
    img = Image.open(image_path)
    width, height = img.size
    cut_width = int(height/cut_divisor)#8)
    cut_height = int(height/cut_divisor)#)
    img = cut_patch(img, cut_width, cut_height, point_x, point_y)
    img = img.resize((crop_width, crop_height), Image.NEAREST)    
    return img

def load_image_and_crop(image_path, point_x, point_y, crop_width, crop_height, cut_divisor=8, images_bucket_name='reefcloud-image-uploads'):
    boto3_session = boto3.session.Session(profile_name='RCPoweruser')
    file_stream = load_s3_im(image_path, boto3_session, images_bucket_name)
    img = Image.open(file_stream)
    #img = Image.open(image_path)

    width, height = img.size
    cut_width = int(height/cut_divisor)#8)
    cut_height = int(height/cut_divisor)#)

    img = cut_patch(img, cut_width, cut_height, point_x, point_y)
    img = img.resize((crop_width, crop_height), Image.NEAREST)

    return img

# as above but uses preloaded image. Used in inference for generating batches so image only needs to be loaded once.
def load_image_and_crop_inf(img, crop_width, crop_height, point_x, point_y, cut_divisor=8):
    width, height = img.size
    cut_width = int(height/cut_divisor)
    cut_height = int(height/cut_divisor)

    img = cut_patch(img, cut_width, cut_height, point_x, point_y)
    img = img.resize((crop_width, crop_height), Image.NEAREST)

    return img


def check_non_existing_images(image_paths, images_bucket_name, still_checking_callback):
    existing_image_paths = []
    missing_image_paths = []
    error_image_paths = []
    boto3.Session()
    s3 = boto3.resource("s3")
    images_bucket = s3.Bucket(images_bucket_name)
    image_check_count = 0
    for image_path in image_paths:
        if image_check_count % 1000 == 0:
            try:
                still_checking_callback.heartbeat_callback(
                    'within check_non_existing_images after checking {} images of {}'.format(image_check_count, len(image_paths)))
            except:
                pass
        try:
            image_bucket_key = image_path.replace('/home/ubuntu/images/', '')
            if not image_path in existing_image_paths \
                    and not image_path in missing_image_paths \
                    and not image_path in error_image_paths:
                objs = list(images_bucket.objects.filter(Prefix=image_bucket_key))
                if len(objs) == 0 or objs[0].key != image_bucket_key:
                    missing_image_paths.append(image_path)
                else:
                    existing_image_paths.append(image_path)
        except:
            error = "Unexpected error checking existence of image_path: " + image_path + ' - ' + str(sys.exc_info())
            print('error - ' + error)
            error_image_paths.append(image_path)
        image_check_count += 1

    return existing_image_paths, missing_image_paths, error_image_paths


def finalValAcc(runpath):
    df = pd.read_csv(os.path.join(runpath,'csvlogs.csv'))
    valAcc = df[df.index==df.val_loss.idxmin()].val_acc.to_list()[0]
    return valAcc


def calculate_mean_image_from_crop_file_list(imlist, imgwidth=256, imgheight=256):#imgwidth=64, imgheight=64):#, imgwidth=256, imgheight=256):

    print("Calculating mean image")

    count = 0
    beginTime = time.time()

    # Access all PNG files in directory
    #imlist = glob2.glob(directory + "/**/*.jpg")
    #imlist = [filename for filename in allfiles if filename[-4:] in [".jpg"]]

    # Assuming all images are the same size, get dimensions of first image
    #w, h = Image.open(imlist[0]).size
    w = imgwidth
    h = imgheight
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:


        img = load_image_and_crop(im["image_path"], im["point_x"], im["point_y"], w,h)#256, 256)

        imarr = np.array(img, dtype=np.float)
        arr = arr + imarr / N

        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - beginTime
            print("Processed {} images in {:.2f} seconds. "
                  "{:.2f} images/second.".format(count, elapsed,
                                                 count / elapsed))

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)
    print (arr.shape)
    return arr


def calculate_mean_image_from_file_list(imlist,imgwidth=256, imgheight=256):#imgwidth=64, imgheight=64):# imgwidth=256, imgheight=256):

    print("Calculating mean image")

    count = 0
    beginTime = time.time()

    # Access all PNG files in directory
    #imlist = glob2.glob(directory + "/**/*.jpg")
    #imlist = [filename for filename in allfiles if filename[-4:] in [".jpg"]]

    # Assuming all images are the same size, get dimensions of first image
    #w, h = Image.open(imlist[0]).size
    w = imgwidth
    h = imgheight
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        img = Image.open(im)
        img = img.resize((w, h), Image.NEAREST)
        imarr = np.array(img, dtype=np.float)
        arr = arr + imarr / N

        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - beginTime
            print("Processed {} images in {:.2f} seconds. "
                  "{:.2f} images/second.".format(count, elapsed,
                                                 count / elapsed))

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)
    print (arr.shape)
    return arr


def calculate_mean_image_all_channels(directory,imgwidth=256, imgheight=256):#imgwidth=64, imgheight=64):# imgwidth=256, imgheight=256):
    exts = ["jpg", "png"]

    mean = np.zeros((1, 3, imgwidth, imgheight))
    mean = np.zeros((imgwidth, imgheight, 3))
    N = 0

    beginTime = time.time()
    for subdir, dirs, files in os.walk(directory):
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in exts):
                img = io.imread(os.path.join(subdir, fName))
                if img.shape == (imgwidth, imgheight, 3):

                    mean[:, :, 0] += img[:, :, 0]
                    mean[:, :, 1] += img[:, :, 1]
                    mean[:, :, 2] += img[:, :, 2]

                    N += 1
                    if N % 1000 == 0:
                        elapsed = time.time() - beginTime
                        print("Processed {} images in {:.2f} seconds. "
                              "{:.2f} images/second.".format(N, elapsed,
                                                             N / elapsed))
    mean[0] /= N

    #return mean.reshape(imgwidth, imgheight, 3)
    return mean


def _get_batches_of_transformed_samples(self, index_array):
    batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
    batch_x2 = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
    batch_x3 = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())

    grayscale = self.color_mode == 'grayscale'
    # build batch of image data
    for i, j in enumerate(index_array):
        '''
        fname = self.filenames[j]

        img = load_img(os.path.join(self.directory, fname),
                       grayscale=grayscale,
                       target_size=self.target_size,
                       interpolation=self.interpolation)
        x = img_to_array(img, data_format=self.data_format)
        x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)

        x = array_to_img(x)
        x = x.crop((64, 64, 192, 192))
        x = x.resize((256, 256))

        if not os.path.isfile("sample.jpg"):
            x.save("sample.jpg")

        x = img_to_array(x, data_format=self.data_format)

        batch_x[i] = x
        '''
        '''
        fname = self.filenames[j]
        img = load_img(os.path.join(self.directory, fname),
                       grayscale=grayscale,
                       target_size=self.target_size)
        x = img_to_array(img, data_format=self.data_format)
        x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)
        batch_x[i] = x
        # print(self.directory + fname)
        fname = self.filenames[j].replace(".8.", ".4.")
        dir = self.directory.replace("train", "train4").replace("test", "test4").replace("val", "val4").replace(
            "the_rest", "the_rest4")
        img = load_img(os.path.join(dir, fname),
                       grayscale=grayscale,
                       target_size=self.target_size)
        x2 = img_to_array(img, data_format=self.data_format)
        x2 = self.image_data_generator.random_transform(x2)
        x2 = self.image_data_generator.standardize(x2)
        batch_x2[i] = x2

        #fname = self.filenames[j].replace(".8.", ".4.")
        #dir = self.directory.replace("train", "train4").replace("test", "test4").replace("val", "val4").replace(
        #    "the_rest", "the_rest4")
        '''

        fname = self.filenames[j].replace(".8.", ".4.")
        dir = self.directory.replace("train", "train4").replace("test", "test4").replace("val", "val4").replace(
            "the_rest", "the_rest4")
        img2 = load_img(os.path.join(dir, fname),
                       grayscale=grayscale,
                       target_size=(512, 512))

        x2 = img_to_array(img2, data_format=self.data_format)
        x2 = self.image_data_generator.random_transform(x2)
        x2 = self.image_data_generator.standardize(x2)

        x2 = array_to_img(x2)
        x1 = x2.crop((128, 128, 384, 384))
        x2 = x2.resize((256, 256))
        '''
        name = fname.split("/")[1]
        if not os.path.isfile(name + ".small.jpg"):
            x1.save(name +".small.jpg")
            x2.save(name +".big.jpg")
        '''
        x2 = img_to_array(x2, data_format=self.data_format)
        x1 = img_to_array(x1, data_format=self.data_format)
        batch_x2[i] = x2
        batch_x[i] = x1

        """
        fname = self.filenames[j]
        dir = self.directory
        img2 = load_img(os.path.join(dir, fname),
                        grayscale=grayscale,
                        target_size=self.target_size)

        x2 = img_to_array(img2, data_format=self.data_format)
        x2 = self.image_data_generator.random_transform(x2)
        x2 = self.image_data_generator.standardize(x2)

        x2 = array_to_img(x2)
        x1 = x2.crop((64, 64, 192, 192))
        x1 = x1.resize((256, 256))

        if not os.path.isfile("small.jpg"):
            x1.save("small.jpg")
            x2.save("big.jpg")

        x2 = img_to_array(x2, data_format=self.data_format)
        x1 = img_to_array(x1, data_format=self.data_format)
        batch_x2[i] = x2
        batch_x[i] = x1
        """
        '''
        fname = self.filenames[j].replace(".8.", ".2.")
        dir = self.directory.replace("train", "train2").replace("test", "test2").replace("val", "val2").replace(
            "the_rest", "the_rest2")
        img3 = load_img(os.path.join(dir, fname),
                        grayscale=grayscale,
                        target_size=self.target_size)
        x3 = img_to_array(img3, data_format=self.data_format)
        x3 = self.image_data_generator.random_transform(x3)
        x3 = self.image_data_generator.standardize(x3)
        batch_x3[i] = x3
        '''
    # optionally save augmented images to disk for debugging purposes
    if self.save_to_dir:
        for i, j in enumerate(index_array):
            img = array_to_img(batch_x[i], self.data_format, scale=True)
            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                              index=j,
                                                              hash=np.random.randint(1e7),
                                                              format=self.save_format)
            img.save(os.path.join(self.save_to_dir, fname))
    # build batch of labels
    if self.class_mode == 'input':
        batch_y = batch_x.copy()
    elif self.class_mode == 'sparse':
        batch_y = self.classes[index_array]
    elif self.class_mode == 'binary':
        batch_y = self.classes[index_array].astype(K.floatx())
    elif self.class_mode == 'categorical':
        batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.
    else:
        return batch_x
    #return [batch_x, batch_x2, batch_x3], batch_y
    #return batch_x, batch_y
    return [batch_x, batch_x2], batch_y


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            ib = ib[0]
            ib2 = ib[1]
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                imgs2 = np.zeros((self.nb_steps * ib2.shape[0], *ib2.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            imgs2[s * ib2.shape[0]:(s + 1) * ib2.shape[0]] = ib2
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
            print("rading val data")
        self.validation_data = [imgs, imgs2, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)



class PatchedModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(PatchedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        
                        saved_correctly = False
                        while not saved_correctly:
                            try:
                                if self.save_weights_only:
                                    self.model.save_weights(filepath, overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                                saved_correctly = True
                            except Exception as error:
                                print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                                sleep(5)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                saved_correctly = False
                while not saved_correctly:
                    try:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        saved_correctly = True
                    except Exception as error:
                        print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                        sleep(5)
