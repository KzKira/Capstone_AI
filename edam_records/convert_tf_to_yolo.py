import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

# Konfigurasi Path
BASE_DIR = "./edam_records"
OUTPUT_DIR = "./yolo_dataset"
SUBSETS = ['train', 'valid', 'test']
IMAGE_SIZE = 224

# Mapping Kelas berdasarkan README
# 0: Fragment, 1: Fiber
CLASS_MAPPING = {
    0: "Fragment",
    1: "Fiber"
}

def decode_label(label: np.ndarray, img_shape: tuple):
    """
    Fungsi ini diambil dan diadaptasi dari how_to_parse.ipynb.
    Mengubah output raw tensor menjadi bounding box piksel absolut.
    """
    label_shape = label.shape

    # Reorder index: dari (y, x, ...) ke (x, y, ...) untuk grid
    label = np.concatenate([label[..., 1:2], label[..., 0:1], label[..., 2:]], axis=-1) 

    grid_coord = np.stack(np.mgrid[:label_shape[0], :label_shape[1]], axis=-1)
    pad = np.zeros((*label_shape[:2], label_shape[-1]-2))
    grid_coord = np.concatenate([grid_coord, pad], axis=-1)
    grid_coord = np.expand_dims(grid_coord, axis=-2)
    grid_coord = np.repeat(grid_coord, label_shape[2], -2)

    # Menambahkan koordinat grid
    label = label + grid_coord 

    grid_height = img_shape[0] / label_shape[0]
    grid_width = img_shape[1] / label_shape[1]

    # Scaler untuk mengembalikan ke ukuran piksel asli (x, y, w, h)
    rewinder = np.array([grid_height, grid_width, img_shape[1], img_shape[0], 1, 1, 1], np.float32)

    label = label * rewinder 

    # Ambil yang confidence-nya 1.0
    # Output format: [x_center, y_center, width, height, confidence, isFragment, isFiber]
    bboxes = label[label[..., 4] == 1.0]
    
    # Pastikan urutan [x, y, w, h, ...]
    if len(bboxes) > 0:
        bboxes = np.concatenate([bboxes[..., 1:2], bboxes[..., 0:1], bboxes[..., 2:]], axis=-1) 
        
    return bboxes

def parse_tfrecord_fn(example):
    """
    Fungsi parsing dari how_to_parse.ipynb
    """
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((16, 16, 6, 7), tf.float32),
    }
    
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))

    label = example["label"]
    label = tf.reshape(label, (16, 16, 6, 7))
    
    return image, label

def convert_to_yolo_format(bboxes, img_width, img_height):
    """
    Mengubah bbox piksel ke format normalisasi YOLO.
    YOLO format: class_id x_center y_center width height
    (Semua nilai x, y, w, h dinormalisasi 0.0 - 1.0)
    """
    yolo_labels = []
    
    for box in bboxes:
        # box structure: [x_pix, y_pix, w_pix, h_pix, conf, isFragment, isFiber]
        x_pix, y_pix, w_pix, h_pix = box[0], box[1], box[2], box[3]
        is_fragment = box[5]
        is_fiber = box[6]
        
        # Tentukan Class ID
        if is_fragment == 1.0:
            class_id = 0
        elif is_fiber == 1.0:
            class_id = 1
        else:
            continue # Skip jika tidak ada kelas yang valid

        # Normalisasi (Sebenarnya data di TFRecord sudah di-scale di fungsi decode, 
        # jadi kita bagi lagi dengan ukuran gambar untuk dapat 0-1)
        x_norm = x_pix / img_width
        y_norm = y_pix / img_height
        w_norm = w_pix / img_width
        h_norm = h_pix / img_height

        # Clipping untuk memastikan nilai tidak lebih dari 1 atau kurang dari 0
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        yolo_labels.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        
    return yolo_labels

def main():
    # Membuat folder output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for subset in SUBSETS:
        print(f"Processing {subset} data...")
        
        # Path untuk menyimpan gambar dan label
        images_dir = os.path.join(OUTPUT_DIR, subset, "images")
        labels_dir = os.path.join(OUTPUT_DIR, subset, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Ambil semua file .tfrecord di folder subset
        tfrecord_pattern = os.path.join(BASE_DIR, subset, "*.tfrecord")
        files = glob.glob(tfrecord_pattern)
        
        if not files:
            print(f"No tfrecord files found in {tfrecord_pattern}")
            continue

        # Load dataset
        raw_dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
        
        # Kita beri nama file secara urut karena TFRecord tidak menyimpan nama file asli
        file_counter = 0
        
        for image, label in tqdm(parsed_dataset):
            # Convert tensor ke numpy
            img_np = image.numpy()
            label_np = label.numpy()
            
            # Decode bounding boxes ke koordinat piksel
            bboxes_pixel = decode_label(label_np, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Convert ke YOLO format (normalized)
            yolo_lines = convert_to_yolo_format(bboxes_pixel, IMAGE_SIZE, IMAGE_SIZE)
            
            # Jika ada objek (yolo_lines tidak kosong), simpan data
            # (Opsional: jika ingin menyimpan background images (tanpa objek), hapus if ini)
            if len(yolo_lines) > 0:
                filename = f"{subset}_{file_counter:05d}"
                
                # 1. Simpan Gambar (.jpg)
                # OpenCV menggunakan BGR, TensorFlow menggunakan RGB. Perlu di-convert.
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(images_dir, f"{filename}.jpg"), img_bgr)
                
                # 2. Simpan Label (.txt)
                with open(os.path.join(labels_dir, f"{filename}.txt"), "w") as f:
                    f.write("\n".join(yolo_lines))
                
                file_counter += 1

    print("Conversion Complete!")
    print(f"Dataset saved to {OUTPUT_DIR}")
    
    # Membuat file data.yaml untuk YOLO
    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['Fragment', 'Fiber']
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created data.yaml for YOLO training.")

if __name__ == "__main__":
    main()