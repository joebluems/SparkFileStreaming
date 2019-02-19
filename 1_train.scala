import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split,CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml.util

//load data , create pipeline , generate train/test
val df = spark.read.option("header", "true").option("inferSchema", "true").csv("sample1000.csv")

def getPipeline( continuousFeatures: Seq[String] ): Pipeline = {
 val assembler = new VectorAssembler().setInputCols((continuousFeatures ).toArray).setOutputCol("features")
 val rf = new RandomForestClassifier().setLabelCol("target").setFeaturesCol("features").
   setMaxBins(500).setNumTrees(100).setSeed(199)
 val pipeline = new Pipeline().setStages(Array(assembler,rf))
 pipeline
}

val continuousFeatures = Seq("feature1","feature2","feature3","feature4","feature5")
val pipeline = getPipeline( continuousFeatures)
val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed=199)


// fit model on train and evaluate on test set
val model = pipeline.fit(train)
val predictions = model.transform(test)
val evaluatorAUROC = new BinaryClassificationEvaluator().
  setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")
val auroc = evaluatorAUROC.evaluate(predictions)

model.write.overwrite.save("testrf")


