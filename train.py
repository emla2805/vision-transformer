from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard

from model import VisionTransformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    ds = tfds.load("cifar10", as_supervised=True)
    ds_train = (
        ds["train"]
        .cache()
        .shuffle(1024)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = ds["test"].batch(32)

    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_classes=10,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        channels=3,
        dropout=0.1,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=[
            TensorBoard(log_dir=args.logdir, profile_batch=0),
        ],
    )
