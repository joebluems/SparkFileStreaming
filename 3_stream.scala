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


/// load schema of file
val schema = StructType(
  Array(StructField("id", StringType),
        StructField("target", IntegerType),
        StructField("feature1", DoubleType),
        StructField("feature2", DoubleType),
        StructField("feature3", DoubleType),
        StructField("feature4", DoubleType),
        StructField("feature5", DoubleType)))

/// load the model pipeline
val model = PipelineModel.read.load("testrf")

/// setup the stream
val fileStreamDf = spark.readStream.
  option("header", "true").
  schema(schema).
  csv("stream_test")

val predictions = model.transform(fileStreamDf)
val evaluatorAUROC = new BinaryClassificationEvaluator().
  setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val auroc = evaluatorAUROC.evaluate(predictions)
