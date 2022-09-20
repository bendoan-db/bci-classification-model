# Databricks notebook source
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import mlflow
from petastorm.spark import SparkDatasetConverter, make_spark_converter

# COMMAND ----------

# DBTITLE 1,Load and Split Dataframe
dataset = spark.read.table(training_df)
train_df,val_df = dataset.randomSplit([0.8, 0.2])

#get num classes 
num_labels = dataset_df.select("label").distinct().count()

train_df = dataset_split[0]
val_df = dataset_split[1]

# COMMAND ----------

# DBTITLE 1,Convert to Tensorflow Datasets
#set cache location for spark converters
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

#instantiate Spark converters for dataset
converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

mlflow.tensorflow.autolog()

#use mirrored strategy to parrellize training
strategy = tf.distribute.MirroredStrategy()

input_shape = 

with strategy.scope():
  # Define model inputs.
  inputs = layers.Input(shape=(100,100,3))
  # Define network shape.
  w = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
  w = layers.Dropout(0.2)(w)
  w = layers.MaxPooling2D(padding='same')(w)
  w = layers.Conv2D(64, 3, activation='relu', padding='same')(w)
  w = layers.Conv2D(64, 3, activation='relu', padding='same')(w)
  w = layers.Dropout(0.2)(w)
  w = layers.MaxPooling2D(padding='same')(w)
  w = layers.Conv2D(128, 3, activation='relu', padding='same')(w)
  w = layers.Conv2D(128, 3, activation='relu', padding='same')(w)
  w = layers.Conv2D(128, 3, activation='relu', padding='same')(w)
  w = layers.Dropout(0.2)(w)
  w = layers.MaxPooling2D(padding='same')(w)
  w = layers.Flatten()(w)
  w = layers.Dropout(0.4)(w)
  w = layers.Dense(256, activation='relu')(w)
  w = layers.Dropout(0.4)(w)
  w = layers.Dense(256, activation='relu')(w)
  w = layers.Dropout(0.4)(w)
  w = layers.Dense(256, activation="relu")(w)
  w = layers.Dense(len(sample_classes), activation="softmax")(w)

  model = Model(inputs=inputs, outputs=w)

  # Print and plot network.
  model.summary()
  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

  # Set the hyperparameters for this model run.
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                      beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                      name='Adam'
                                      ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
      )

  # Define the early stopping criteria.
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)

  # Start the training. This will produce a preliminary (course) model.
  h = model.fit(train_dataset,
              validation_data=validation_dataset,
              epochs=25,
              verbose=1,
              callbacks=[es]
      )
