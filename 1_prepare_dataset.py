import os, shutil, random
import xml.etree.ElementTree as ET

# Step 1: VOC XML → YOLO .txt
def convert_voc_to_yolo(xml_dir, label_dir, classes_list=['WBC', 'RBC', 'Platelets']):
    os.makedirs(label_dir, exist_ok=True)
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'): continue
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        # 1 - Obtain image size
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        # 2 - Convert each object to YOLO format
        yolo_lines = []
        for obj in root.iter('object'):
            obj_class = obj.find('name').text
            if obj_class not in classes_list: continue
            class_id = classes_list.index(obj_class) 
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            box_to_yolo = ((box[0]+ box[1]) / (2 * w), (box[2] + box[3]) / (2 * h), (box[1] - box[0]) / w, (box[3] - box[2]) / h)
            yolo_lines.append(f"{class_id} {' '.join(map(str, box_to_yolo))}")
        with open(os.path.join(label_dir, xml_file.replace('.xml', '.txt')), 'w') as f:
            f.write('\n'.join(yolo_lines))

# Step 2: Split dataset and copy images/labels
def split_and_copy(image_dir, label_dir, output_dir, random_seed=42, split_ratio=(0.7, 0.2, 0.1)):
    random.seed(random_seed)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files_base_name = [os.path.splitext(f)[0] for f in image_files]
    random.shuffle(image_files_base_name)
    n = len(image_files_base_name)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    splits = {'train': image_files_base_name[:n_train], 'val': image_files_base_name[n_train:n_train + n_val], 'test': image_files_base_name[n_train + n_val:]}
    for split, names in splits.items():
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        for name in names:
            img_src = os.path.join(image_dir, name + '.jpg')
            label_src = os.path.join(label_dir, name + '.txt')
            img_dst = os.path.join(output_dir, 'images', split, name + '.jpg')
            label_dst = os.path.join(output_dir, 'labels', split, name + '.txt')
            if os.path.exists(img_src):
                shutil.copy2(img_src, img_dst)
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)

# Step 3: Generate YAML file for YOLO dataset
def generate_yaml(save_dir):
    yaml_path = os.path.join(save_dir, 'bccd.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""# BCCD dataset for YOLO format
                    path: {save_dir}
                    train: images/train
                    val: images/val
                    test: images/test

                    names:
                    0: WBC
                    1: RBC
                    2: Platelets
                    """)

if __name__ == '__main__':
    VOC_ROOT = './BCCD_Dataset/BCCD'
    OUTPUT_DIR = './data/BCCD-YOLO'
    CLASSES = ['WBC', 'RBC', 'Platelets']
    SPLIT_RATIO = (0.7, 0.2, 0.1)  # 70% train, 20% val, 10% test
    RANDOM_SEED = 42 # For reproducibility
    
    xml_dir = os.path.join(VOC_ROOT, 'Annotations')
    image_dir = os.path.join(VOC_ROOT, 'JPEGImages')
    label_dir = os.path.join(VOC_ROOT, 'labels')

    convert_voc_to_yolo(xml_dir, label_dir, CLASSES)
    split_and_copy(image_dir, label_dir, OUTPUT_DIR, random_seed=RANDOM_SEED, split_ratio=SPLIT_RATIO)
    generate_yaml(OUTPUT_DIR)
    