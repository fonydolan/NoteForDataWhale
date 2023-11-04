import argparse
import tensorflow as tf
import os

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--data_dir", default="/gemini/data-1")
parser.add_argument("--train_dir", default="/gemini/output")
args = parser.parse_args()


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [150, 150]) / 255.0
    return image_resized, label


if __name__ == "__main__":
    train_dir = args.data_dir + "/train"
    cats = []
    dogs = []
    for file in os.listdir(train_dir):
        if file.startswith("dog"):
            dogs.append(train_dir + "/" + file)
        else:
            cats.append(train_dir + "/" + file)
    print("dogSize:%d catSize:%d" % (len(cats), len(dogs)))
    train_cat_filenames = tf.constant(cats[:10000])
    train_dog_filenames = tf.constant(dogs[:10000])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_cat_filenames.shape, dtype=tf.int32),
        tf.ones(train_dog_filenames.shape, dtype=tf.int32)
    ], axis=-1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

    train_dataset = train_dataset.map(map_func=_decode_and_resize,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.shuffle(buffer_size=20000)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(train_dataset, epochs=args.num_epochs)
    model.save(args.train_dir)

    # 构建测试数据集
    test_cat_filenames = tf.constant(cats[10000:])
    test_dog_filenames = tf.constant(dogs[10000:])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
    test_labels = tf.concat([
        tf.zeros(test_cat_filenames.shape, dtype=tf.int32),
        tf.ones(test_dog_filenames.shape, dtype=tf.int32)
    ], axis=-1)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(args.batch_size)
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for images, label in test_dataset:
        y_pred = model.predict(images)
        sparse_categorical_accuracy.update_state(y_true=label, y_pred=y_pred)
    print("test accuracy:%f" % sparse_categorical_accuracy.result())
