import argparse
import tensorflow as tf
import os
import json

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument("--num_epochs", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--data_dir", default="/gemini/data-1")
parser.add_argument("--train_dir", default="/gemini/output")
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--rank", default=0, type=int)
args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.list_physical_devices(device_type='GPU')
num_workers = args.num_workers
workList = []
for i in range(num_workers):
    work = os.environ.get("GEMINI_IP_taskrole1_%d" % i) + ":" + os.environ.get(
        "GEMINI_taskrole1_%d_http_PORT" % i)
    workList.append(work)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': workList
    },
    'task': {'type': 'worker', 'index': args.rank}
})
print("TF_CONFIG:", os.environ.get("TF_CONFIG"))
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [224, 224]) / 255.0
    return image_resized, label


if __name__ == "__main__":
    if len(gpus) != 0:
        BATCH_SIZE_PER_REPLICA = args.batch_size * len(gpus)
    else:
        BATCH_SIZE_PER_REPLICA = args.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * args.num_workers
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
    train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    steps_per_epoch = tf.math.ceil(20000 / BATCH_SIZE).numpy()  # 算出step的真实数量
    print("=====================================")
    print("steps_per_epoch:", steps_per_epoch)
    print("batch_size:", BATCH_SIZE)
    print("epoch:", args.num_epochs)
    print("gpus:", gpus)
    print("=====================================")
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(224, 224, 3)),
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
        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory=args.train_dir, checkpoint_name="model.ckpt",
                                             max_to_keep=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )
        model.fit(train_dataset, epochs=args.num_epochs, steps_per_epoch=steps_per_epoch)
    if args.rank == 0:
        print("save model")
        manager.save()
