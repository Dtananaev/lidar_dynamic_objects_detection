import glob
import os
import numpy as np
from detection_3d.tools.file_io import load_bboxes, save_bboxes_to_file


def remove_semicols():
    boxes_dir = (
        "/home/denis/lidar_dynamic_objects_detection/detection_3d/inference/bboxes"
    )
    sequences = glob.glob(boxes_dir + "/*")
    for seq in sequences:
        print(f"seq {seq}")
        seq_number = seq.split("/")[-1]
        box_path = sorted(glob.glob(seq + "/*.txt"))
        for box_file in box_path:

            box = load_bboxes(box_file, False)
            centroid = box[:, :3]
            width = box[:, 3]
            length = box[:, 4]
            height = box[:, 5]
            yaw = box[:, 6]
            label = box[:, 7]
            # filename = "test.txt"
            save_bboxes_to_file(
                box_file, centroid, width, length, height, yaw, label, delim=" "
            )


remove_semicols()
