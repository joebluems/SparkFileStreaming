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
val evalAUC = new BinaryClassificationEvaluator().setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val model_rf = PipelineModel.read.load("testrf")
val model_mlp = PipelineModel.read.load("testmlp")

// score data and calculate AUC - random forest
val pred_rf = model_rf.transform(df)
val auc_rf = evalAUC.evaluate(pred_rf)

// score data and calculate AUC - neural net
val pred_mlp = model_mlp.transform(df)
val auc_mlp = evalAUC.evaluate(pred_mlp)
