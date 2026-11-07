#!/bin/bash
# Write the image filenames of every SAIAB split to splits/<subcollection>_<split>.txt
# Usage: bash make_split_lists.sh /path/to/datasets   (folder that holds aldabra_hbb, tanganyika_hbb, ...)
DATA_DIR="${1:?give the datasets folder}"
mkdir -p splits
for sub in aldabra tanganyika pondoland protea; do
  for split in train valid test; do
    ls "$DATA_DIR/${sub}_hbb/$split/images" | sort > "splits/${sub}_${split}.txt"
  done
done
wc -l splits/*.txt   # should total 12,466 (e.g. tanganyika: 1790 / 518 / 267)
