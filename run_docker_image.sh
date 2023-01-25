#!/bin/bash

docker run -it --mount type=bind,source="$(pwd)",target=/app  --gpus all reefscan-inference /bin/bash -c "cd app/src && python3 reefscan_inference.py"