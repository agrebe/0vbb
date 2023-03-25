#!/bin/bash

# This is a wrapper script to generate the 3-point and 4-point functions

# Call the Python contractions script
python3 contractions.py > python-output

# Post-process the outputs to change products of props
# into calls to tensor contractions
./process-traces.sh
