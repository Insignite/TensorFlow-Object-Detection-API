import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# To prevent python can't detect object_detection
sys.path.append('N:\\TensorFlow\\models\\research\\object_detection')

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path define section
CURRENT_PATH = os.getcwd()
TEST_IMAGE = os.path.join(CURRENT_PATH,'evals','test1.jpg')
FROZEN_GRAPH_CKPT = os.path.join(CURRENT_PATH, 'trained-inference-graphs', 'output_inference_graph_v1.pb','frozen_inference_graph.pb')
LABEL_MAP = os.path.join(CURRENT_PATH, 'annotations', 'label_map.pbtxt')

# Hold variable for number of classes
NUM_CLASSES = 6

# Load label map
labels_map  = label_map_util.load_labelmap(LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(labels_map, max_num_classes=NUM_CLASSES, use_display_name=True)
categories_index = label_map_util.create_category_index(categories)


# Load Tenforflow frozen_inference_graph into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

#Define input and output tensor for the object detection classifier

#Input: image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

#Output: detection box, scores, and classes

# Detection box:
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Scores
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
# Class
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


#Load image with OpenCV
# Expanding image dimensions to shape [1, None, None, 3]
image = cv2.imread(TEST_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

#Perform the actual detection
(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

#Visualizing the result
vis_util.visualize_boxes_and_labels_on_image_array(image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), categories_index,
    use_normalized_coordinates=True, line_thickness=6)

# Display the image
cv2.imshow('Image Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
