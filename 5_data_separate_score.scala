import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.clustering.{KMeans,KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier,MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split,CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml.util


///////////// D A T A  P R E P A R A T I O N //////////
//load data get ready for scoring
val df = spark.read.option("header", "true").option("inferSchema", "true").csv("sample1000.csv")
val continuousFeatures = Seq("feature1","feature2","feature3","feature4","feature5")
val assembler = new VectorAssembler().setInputCols((continuousFeatures ).toArray).setOutputCol("features")
val model_score = assembler.transform(df)

/// load models 
val model_rf = RandomForestClassificationModel.read.load("testrf")
val model_mlp = MultilayerPerceptronClassificationModel.read.load("testmlp")
val model_kmeans = KMeansModel.read.load("testkmeans")

//////////////// E V A L U A T I O N /////////
/// load evaluators
val evaluatorAUROC = new BinaryClassificationEvaluator().setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val evaluator = new ClusteringEvaluator()

// score data with models
val pred_rf = model_rf.transform(model_score)
val pred_mlp = model_mlp.transform(model_score)
val pred_kmeans = model_kmeans.transform(model_score)

// calculate metrics
val auc_rf = evaluatorAUROC.evaluate(pred_rf)
val auc_mlp = evaluatorAUROC.evaluate(pred_mlp)
val silhouette = evaluator.evaluate(pred_kmeans)


