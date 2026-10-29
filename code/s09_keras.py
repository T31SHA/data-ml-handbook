# HIDE
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ENDHIDE
import numpy as np, tensorflow as tf
from tensorflow import keras
from keras import layers
keras.utils.set_random_seed(0)
X = np.random.randn(4000, 20).astype("float32"); w = np.random.randn(20, 3)
y = (X @ w + 0.5 * np.random.randn(4000, 3)).argmax(1)
train = tf.data.Dataset.from_tensor_slices((X[:3200], y[:3200])).shuffle(3200).batch(64).prefetch(tf.data.AUTOTUNE)
val = tf.data.Dataset.from_tensor_slices((X[3200:], y[3200:])).batch(256)
model = keras.Sequential([layers.Input((20,)), layers.Dense(64), layers.LayerNormalization(),
                          layers.Activation("gelu"), layers.Dropout(0.2), layers.Dense(3)])     # logits
model.compile(optimizer=keras.optimizers.AdamW(3e-3, weight_decay=1e-2),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
hist = model.fit(train, validation_data=val, epochs=30, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
h = hist.history
for e in range(0, len(h["loss"]), 5): print(f"epoch {e:2d} loss={h['loss'][e]:.3f} val_loss={h['val_loss'][e]:.3f} val_acc={h['val_accuracy'][e]:.3f}")
print(f"stopped after {len(h['loss'])} epochs, best val_loss={min(h['val_loss']):.3f}, params={model.count_params()}")
