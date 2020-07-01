from PIL import Image
import glob
import os
import numpy as np


def concat_images():
    images_dir = "/home/denis/lidar_dynamic_objects_detection/detection_3d/inference/image_bboxes"

    out_path = (
        "/home/denis/lidar_dynamic_objects_detection/detection_3d/inference/concat"
    )
    sequences = glob.glob(images_dir + "/*")
    print(f"sequences {sequences}")
    for seq in sequences:
        print(f"seq {seq}")
        seq_number = seq.split("/")[-1]
        out_dir = os.path.join(out_path, seq_number)
        os.makedirs(out_dir, exist_ok=True)
        front_cam_path = sorted(glob.glob(seq + "/front_camera/*.png"))
        print(f"front_cam_path {front_cam_path}")
        front_left_cam_path = sorted(glob.glob(seq + "/front_left_camera/*.png"))
        front_right_cam_path = sorted(glob.glob(seq + "/front_right_camera/*.png"))
        for front, front_left, front_right in zip(
            front_cam_path, front_left_cam_path, front_right_cam_path
        ):
            name = os.path.splitext(os.path.basename(front))[0]
            front_img = np.asarray(Image.open(front))
            left_img = np.asarray(Image.open(front_left))
            right_img = np.asarray(Image.open(front_right))

            img_concat = np.hstack((left_img, front_img, right_img))
            img = Image.fromarray(img_concat.astype("uint8"))
            img_out_name = os.path.join(out_dir, name + ".png")
            img.save(img_out_name)


concat_images()
