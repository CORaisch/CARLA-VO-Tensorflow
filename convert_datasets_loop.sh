#!/bin/bash

NUMS=(00 01 02 03 04 05 06 07 08 09 10)
for NUM in "${NUMS[@]}"
do
    python sequence_to_tfrec.py /media/claudio/1AC5-C2D4/Datasets/KITTI/TFRecords/${NUM}.zip /media/claudio/1AC5-C2D4/Datasets/KITTI/gt_poses/converted-to-CARLA/euler/${NUM}_carla_euler.txt '/media/claudio/1AC5-C2D4/Datasets/KITTI/sequences_gray/'${NUM}'/image_0=rgb_left' -imc 1 -imw 1216 -imh 320
done
