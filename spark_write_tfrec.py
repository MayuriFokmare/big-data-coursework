
import tensorflow as tf
import os

# Initialize public dataset and cloud storage bucket name
GCS_PATTERN = 'gs://cloud-samples-data/ai-platform/flowers_tfrec/*/*.jpg'
GCS_OUTPUT = 'gs://big-data-coursework-457710-storage/tfrecords-spark/'

# Using 2% images for faster testing
SAMPLE_FRACTION = 0.02

# Initialize SparkContext
from pyspark import SparkContext
sc = SparkContext.getOrCreate()

# Constants
TARGET_SIZE = [192, 192]
CLASSES = [b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips']

# Mapping functions
# Read JPEG file and extract its image tensor and label (folder name)
def decode_jpeg_and_label(filepath):
    bits = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.split(tf.expand_dims(filepath, axis=-1), sep='/')
    label2 = label.values[-2]
    return image, label2

# Resize image to target size while preserving aspect ratio and center crop
def resize_and_crop_image_spark(image_label_tuple):
    image, label = image_label_tuple
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw, th = TARGET_SIZE[1], TARGET_SIZE[0]
    resize_crit = (w * th) / (h * tw)
    
    # Resize image based on aspect ratio:
    # Width-first if too tall, height-first if too wide
    image = tf.cond(
        resize_crit < 1,
        lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),
        lambda: tf.image.resize(image, [w * th / h, h * th / h])
    )
    
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    
    # Center crop the resized image to the target size
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label

# Compress image to JPEG format and convert to bytes for TFRecord writing
def recompress_image_spark(image_label_tuple):
    image, label = image_label_tuple
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    
    # Convert to bytes
    image = image.numpy()
    return image, label

# Create a bytes_list feature from byte strings
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

# Create an int64_list feature from integers
def _int_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

# Create a tf.train.Example containing image bytes and label
def to_tfrecord(img_bytes, label):
    class_num = int(tf.argmax(tf.constant(CLASSES) == label))
    feature = {
        "image": _bytestring_feature([img_bytes]),
        "class": _int_feature([class_num])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Step 1: List all filenames matching the GCS pattern
filenames = tf.io.gfile.glob(GCS_PATTERN)

# Step 2: Create RDD from the list of filenames
filenames_rdd = sc.parallelize(filenames)

# Step 3: Sample 2% randomly
sampled_filenames_rdd = filenames_rdd.sample(False, SAMPLE_FRACTION)

# Step 4: Decode images
decoded_rdd = sampled_filenames_rdd.map(lambda filepath: decode_jpeg_and_label(filepath))

# Step 5: Resize and crop images
resized_rdd = decoded_rdd.map(resize_and_crop_image_spark)

# Step 6: Recompress images to JPEG bytes
recompressed_rdd = resized_rdd.map(recompress_image_spark)

# Define TFRecord writer function
def write_partition(partition_idx, iterator):
    # Write each partition's (image, label) pairs into a separate TFRecord file.
    output_path = GCS_OUTPUT + f"partition-{partition_idx:02d}.tfrec"
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_bytes, label in iterator:
            # Convert (image bytes, label) into a tf.train.Example
            example = to_tfrecord(img_bytes, label)
            writer.write(example.SerializeToString())
    yield output_path

# Step 7: Write out TFRecord files
output_files = recompressed_rdd.mapPartitionsWithIndex(write_partition).collect()

# Step 8: Print output file names
print("TFRecord files created:")
for f in output_files:
    print(f)
