#!/bin/bash

/isaac-sim/kit/python/bin/python3 -m pip install \
    "numpy>=1.23,<2.0" \
    scipy \
    --upgrade

/isaac-sim/kit/python/bin/python3 -m pip install \
    "numpy>=1.23,<2.0" \
    --target /root/.local/share/ov/data/Kit/Isaac-Sim/4.1/pip3-envs/default/ \
    --upgrade