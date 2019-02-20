import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import scala.collection.mutable.Builder
import org.apache.spark.ml._

//load data , load  pipeline , score data
val df = spark.read.option("header", "true").option("inferSchema", "true").csv("sample1000.csv")
val evalAUC = new BinaryClassificationEvaluator().setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val evaluator = new ClusteringEvaluator()
val model_rf = PipelineModel.read.load("testrf")
val model_mlp = PipelineModel.read.load("testmlp")
val model_kmeans = PipelineModel.read.load("testkmeans")

// score data and calculate AUC - random forest
val pred_rf = model_rf.transform(df)
val auc_rf = evalAUC.evaluate(pred_rf)

// score data and calculate AUC - neural net
val pred_mlp = model_mlp.transform(df)
val auc_mlp = evalAUC.evaluate(pred_mlp)

// cluster data and calculate MSE
val pred_kmeans = model_kmeans.transform(df)
val silhouette = evaluator.evaluate(pred_kmeans)
