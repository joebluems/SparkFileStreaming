package com.mapr.streamtest 

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.{OutputMode, Trigger}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split,CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml._
import org.apache.spark.sql.types.{StringType, IntegerType, DoubleType,StructField, StructType}

object Main extends App { 

    val sparkSession = SparkSession.builder
      .master("local")
      .appName("example")
      .getOrCreate()

    val schema = StructType(
      Array(StructField("id", StringType),
        StructField("target", IntegerType),
        StructField("feature1", DoubleType),
        StructField("feature2", DoubleType),
        StructField("feature3", DoubleType),
        StructField("feature4", DoubleType),
        StructField("feature5", DoubleType)))

    /// load the model pipeline
    val model = PipelineModel.read.load("/Users/joeblue/nodejs/streaming/testrf")

    //create stream from folder
    val fileStreamDf = sparkSession.readStream
      .option("header", "true")
      .schema(schema)
      .csv("/Users/joeblue/nodejs/streaming/stream_test")

    val query = fileStreamDf.writeStream
      .format("console")
      .outputMode(OutputMode.Append()).start()

    query.awaitTermination()



}
