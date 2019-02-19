import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split,CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml._

//load data , load  pipeline , score data
val df = spark.read.option("header", "true").option("inferSchema", "true").csv("sample1000.csv")
val model = PipelineModel.read.load("testrf")
val predictions = model.transform(df)

val evaluatorAUROC = new BinaryClassificationEvaluator().
  setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val auroc = evaluatorAUROC.evaluate(predictions)
