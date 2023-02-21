#!/bin/bash

date

python3 contractions.py > python-output
./process-traces.sh

date
