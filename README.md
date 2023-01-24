# reefscan_inference
This application runs inference on a folder of images and exports a .csv file containing the following data fields
```
image_path | image_id | point_num | point_id | point_coordinate | true_class | pred_class | true_group | pred_group
```

Additionally, it produces another .csv file containing a coverage summary of the class groups.

## Running the program
```
python3 reefscan_inference.py <optional arguments>
```

Optional arguments along with their default values:
```
--feature_extractor='../models/ft_ext/weights.best.hdf5' 
--classifier='../models/classifier/reefscan.sav'
--group_labels_csv_file='../models/reefscan_group_labels.csv'
--points_csv_file='../data/reefscan_points.csv'
--local_image_dir='../data/input_images'
--image_path_key='image_path'
--point_x_key='pointx'
--point_y_key='pointy' 
--label_key='point_human_classification'
--cut_divisor=12
--intermediate_feature_outputs_path='../models/features.csv'
--output_results_file='../results/results.csv'
--output_coverage_file='../results/coverage-summary.csv'
```

## Usage

If you are using the program with the default arguments, do the following steps. Otherwise, substitute the arguments with the specified path as needed.
1. Place the input images inside `./data/input_images`
2. Replace the file `./data/reefscan_points.csv` with the one corresponding to the images
3. Run the program
    - The program will load `./models/ft_ext/weights.best.hdf5` as the feature extraction model
    - The program will load `./models/classifier/reefscan.sav` as the classification model
4. The program will extract an intermediate `./models/features.csv` file that acts as a cache so that feature extraction is only done once for the particular set of images to be inferenced.
5. The program will output `./results.csv` and `./group-coverage.json` for the results.


# Docker image

There is a prebuilt Linux docker image that contains all the dependencies required for running this application, e.g. CUDA toolkit, CUDNN.  

It can be found [here](https://aimsgovau-my.sharepoint.com/personal/p_tenedero_aims_gov_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fp%5Ftenedero%5Faims%5Fgov%5Fau%2FDocuments%2Freefscan%2Dinference%2Dfiles%2Freefscan%2Dinference%2Dlatest%2Etar%2Egz&parent=%2Fpersonal%2Fp%5Ftenedero%5Faims%5Fgov%5Fau%2FDocuments%2Freefscan%2Dinference%2Dfiles). 

Note: It is a huge file (~5GB compressed / ~10GB on disk).

Load it via the command `docker load --input reefscan-inference-latest.tar.gz`

Otherwise, it can be built locally if it is more convenient.

## Requirements for Windows
- wsl2
    - https://learn.microsoft.com/en-us/windows/wsl/install ,  or
    - run the following commands in Windows Powershell
        ```
        wsl --install -d Ubuntu
        wsl --setdefault Ubuntu
        wsl --set-version Ubuntu 2
        ```

## Building the docker image

**NOTE: If you are using the prebuilt docker image, skip this step and navigate to `Running the docker image`**

Within wsl2, run:
```
docker build -t reefscan-inference .
```

## Running the docker image
```
docker run --gpus all reefscan-inference /bin/bash -c "cd app/src && python3 reefscan_inference.py"
```

Alternatively, run the script `run_docker_image.sh`

### Additional arguments
Simply add the additional arguments at the end of the command
```
docker run --gpus all reefscan-inference /bin/bash -c "cd app/src && python3 reefscan_inference.py <optional arguments>"
```


