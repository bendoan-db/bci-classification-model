# Databricks notebook source
raw_wav = spark.read.table("hive_metastore.bat_stuff.bat_call_wav_files")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

species_frequencies = spark.read.table("hive_metastore.bat_stuff.species_frequencies")

# COMMAND ----------


