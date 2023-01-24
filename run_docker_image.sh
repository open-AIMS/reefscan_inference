#!/bin/bash

docker run --gpus all reefscan-inference /bin/bash -c "cd app/src && python3 reefscan_inference.py"