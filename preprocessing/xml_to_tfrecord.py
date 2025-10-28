import os
import glob
import pandas as pd
import tensorflow as tf
from lxml import etree
from object_detection.utils import dataset_util

# Path settings
IMAGE_DIR = 'dataset/images'
ANNOTATION_DIR = 'dataset/annotations'
OUTPUT_DIR = 'dataset/'

LABELS = {'بیکٹیریل بلیٹ': 1}  # Update if you have more diseases

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = etree.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def class_text_to_int(row_label):
    return LABELS.get(row_label, None)

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    filename = group.filename.encode('utf8')
    width = int(group.width)
    height = int(group.height)

    xmins = [group.xmin / width]
    xmaxs = [group.xmax / width]
    ymins = [group.ymin / height]
    ymaxs = [group.ymax / height]
    classes_text = [group['class'].encode('utf8')]
    classes = [class_text_to_int(group['class'])]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_tfrecord(image_dir, annotation_dir, output_path):
    xml_df = xml_to_csv(annotation_dir)
    writer = tf.io.TFRecordWriter(output_path)
    for index, row in xml_df.iterrows():
        tf_example = create_tf_example(row, image_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f'Successfully created TFRecord: {output_path}')

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_tfrecord('dataset/images/train', 'dataset/annotations/train', 'dataset/train.record')
    generate_tfrecord('dataset/images/val', 'dataset/annotations/val', 'dataset/val.record')
