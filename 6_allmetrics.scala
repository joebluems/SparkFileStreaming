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
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


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

// score data with models
val pred_rf = model_rf.transform(model_score)
val pred_mlp = model_mlp.transform(model_score)
val pred_kmeans = model_kmeans.transform(model_score)


// evaluation metrics
// 1. Area under ROC
// 2. Area under Precision Recall
// 3. Max F1 value
// 4. Max KS value
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions._
val vectorToColumn = udf{ (x:DenseVector, index: Int) => x(index) }
val toDouble = udf {x: Int => x.toDouble}

////////////// metrics for random forest
// transform scores to RDD, prepare to extract metrics
val score_rf = pred_rf.
   withColumn("score",vectorToColumn(col("probability"),lit(1))).
   withColumn("label",toDouble(col("target"))).select("score","label").
   rdd.map(r => (r.getDouble(0),r.getDouble(1)))
val metrics = new BinaryClassificationMetrics(score_rf)
val f1Score = metrics.fMeasureByThreshold
val roc = metrics.roc.map(r=> (r._1,r._2,r._2-r._1))

/// calculate metrics
val auROC = metrics.areaUnderROC
val auPRC = metrics.areaUnderPR
val maxF1 = f1Score.reduce((x, y) => if(x._2 > y._2) x else y)
val maxKS = roc.reduce((x, y) => if(x._3 > y._3) x else y)


