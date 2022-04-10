#!/bin/bash
module load python3

for i in $(ls ../test-results)
do
  # determine how many operators inserted (2pt, 3pt, or 4pt)
  ops=$(echo $i | cut -d "-" -f 2 | cut -d "p" -f 1)
  python3 compare-results.py $i $ops
done
