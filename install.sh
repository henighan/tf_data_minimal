#!/bin/bash
which -s virtualenv || pip install virtualenv --user
python3 -m virtualenv .work
source .work/bin/activate
python -m pip install -r requirements.txt
python -m pip install --upgrade pip
