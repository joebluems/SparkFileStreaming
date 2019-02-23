import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split,CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml.util


///////////// D A T A  P R E P A R A T I O N //////////
//load data , create pipeline , generate train/test
val df = spark.read.option("header", "true").option("inferSchema", "true").csv("sample1000.csv")
val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed=199)
val continuousFeatures = Seq("feature1","feature2","feature3","feature4","feature5")
val assembler = new VectorAssembler().setInputCols((continuousFeatures ).toArray).setOutputCol("features")
val model_train = assembler.transform(train)
val model_test = assembler.transform(test)


//////////////// T R A I N I N G /////////////
/// train RF
val rf = new RandomForestClassifier().
     setLabelCol("target").setFeaturesCol("features").
     setMaxBins(500).setNumTrees(100).setSeed(199)
val model_rf = rf.fit(model_train)
model_rf.write.overwrite.save("testrf")

// neural net
val layers = Array[Int](5, 3, 3, 2)
val mlp = new MultilayerPerceptronClassifier().
     setLabelCol("target").setFeaturesCol("features").
     setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
val model_mlp = mlp.fit(model_train)
model_mlp.write.overwrite.save("testmlp")

// kmeans
val kmeans = new KMeans().setK(5).setFeaturesCol("features").setPredictionCol("prediction")
val model_kmeans = kmeans.fit(model_train)
model_kmeans.write.overwrite.save("testkmeans")


//////////////// E V A L U A T I O N /////////
/// load evaluators
val evaluatorAUROC = new BinaryClassificationEvaluator().setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val evaluator = new ClusteringEvaluator()

// score test set with models
val pred_rf = model_rf.transform(model_test)
val pred_mlp = model_mlp.transform(model_test)
val pred_kmeans = model_kmeans.transform(model_test)

// calculate metrics
val auc_rf = evaluatorAUROC.evaluate(pred_rf)
val auc_mlp = evaluatorAUROC.evaluate(pred_mlp)
val silhouette = evaluator.evaluate(pred_kmeans)


