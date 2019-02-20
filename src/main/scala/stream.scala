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
import org.apache.log4j.Logger
import org.apache.log4j.Level


object Main extends App { 
    Logger.getLogger("org").setLevel(Level.WARN)

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
    val model_rf = PipelineModel.read.load("/Users/joeblue/nodejs/streaming/testrf")
    val model_mlp = PipelineModel.read.load("/Users/joeblue/nodejs/streaming/testmlp")
    val evalAUC = new BinaryClassificationEvaluator().setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")

    //create stream from folder
    val inputDF = sparkSession.readStream
      .option("header", "true")
      .schema(schema)
      .csv("/Users/joeblue/nodejs/streaming/stream_test")

    // apply random forest 
    val name_rf = "RandFor_feb22"
    val pred_rf = model_rf.transform(inputDF)
    //val auc_rf = evalAUC.evaluate(pred_rf)

    // apply MLP 
    val name_mlp = "Neural_feb22"
    val pred_mlp = model_mlp.transform(inputDF)
    //val auc_mlp = evalAUC.evaluate(pred_mlp)


    // write final DF as stream
    val query = pred_rf.writeStream.format("console").outputMode(OutputMode.Append()).start()
    query.awaitTermination()

}
