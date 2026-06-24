# reefscan_inference
This application runs inference on a folder of images and exports a .csv file containing the following data fields
```
image_path | image_id | point_num | point_id | point_coordinate | true_class | pred_class | true_group | pred_group
```

Additionally, it produces another .csv file containing a coverage summary of the class groups.

## Running the program

First, activate the virtual environment (once per terminal session):
```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Then run from the project root:
```
python inferencer/reefscan_inference.py <optional arguments>
```

Or navigate into the `inferencer` directory and run directly:
```
cd inferencer
python reefscan_inference.py <optional arguments>
```

Optional arguments along with their default values (paths are relative to the current working directory):
```
--feature_extractor='../models/ft_ext/weights.best.hdf5' 
--classifier='../models/classifier/reefscan.sav'
--group_labels_csv_file='../models/reefscan_group_labels.csv'
--points_csv_file=''
--local_image_dir='../data/input_images'
--image_path_key='image_path'
--point_x_key='pointx'
--point_y_key='pointy' 
--label_key='point_human_classification'
--cut_divisor=12
--intermediate_feature_outputs_path='../models/features.csv'
--output_results_file='../results/results.csv'
--output_coverage_file='../results/coverage-summary.csv'
--use_cache='False'
--use_gpu='False'
```

## Usage

1. Activate the virtual environment (see above).
2. Place the input images inside `./data/input_images`, or specify the path explicitly:
```py
    python inferencer/reefscan_inference.py --local_image_dir='<path-to-the-dir-with-new-images>'
```
3. Run the program, overriding the model paths to point at `inferencer/models/`:
```powershell
python inferencer/reefscan_inference.py `
    --feature_extractor=inferencer/models/ft_ext/weights.best.hdf5 `
    --classifier=inferencer/models/classifier/reefscan.sav `
    --group_labels_csv_file=inferencer/models/reefscan_group_labels.csv `
    --intermediate_feature_outputs_path=inferencer/models/features.csv `
    --local_image_dir=data/input_images `
    --output_results_file=results/results.csv `
    --output_coverage_file=results/coverage-summary.csv
```
    - **OPTIONAL: When running the program again for the same set of images, add `--use_cache=True` to skip the feature extraction step. The cached features are stored in `inferencer/models/features.csv`.**
4. The program will output `results/results.csv` for the predictions and `results/coverage-summary.csv` for the coverage information.


(**Update 2023-02-06**: Previously it was required to have a points csv file that goes with the input images. From this date it is now optional. The program will instead internally fill in a new dataframe that contains the 5 points that you would typically see in a points csv file)

# Docker image

There is a prebuilt Linux docker image that contains all the dependencies required for running this application, e.g. CUDA toolkit, CUDNN.  

It can be found [here](https://aimsgovau-my.sharepoint.com/personal/p_tenedero_aims_gov_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fp%5Ftenedero%5Faims%5Fgov%5Fau%2FDocuments%2Freefscan%2Dinference%2Dfiles%2Freefscan%2Dinference%2Dlatest%2Ddockerimg%2Etar%2Egz&parent=%2Fpersonal%2Fp%5Ftenedero%5Faims%5Fgov%5Fau%2FDocuments%2Freefscan%2Dinference%2Dfiles). 

Note: It is a huge file (~5GB compressed / ~10GB on disk).

Load it via the command `docker load --input reefscan-inference-latest-dockerimg.tar.gz`

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
docker run -it --mount type=bind,source="$(pwd)",target=/app --gpus all reefscan-inference /bin/bash -c "cd app/src && python3 reefscan_inference.py"
```

Alternatively, run the script `run_docker_image.sh`

### Additional arguments
Simply add the additional arguments at the end of the command
```
docker run --gpus all reefscan-inference /bin/bash -c "cd app/src && python3 reefscan_inference.py <optional arguments>"
```


