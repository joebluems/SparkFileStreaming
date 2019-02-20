import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split,CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml.util

//load data , create pipeline , generate train/test
val df = spark.read.option("header", "true").option("inferSchema", "true").csv("sample1000.csv")

/// random forest pipeline
def getRandomPipe( continuousFeatures: Seq[String] ): Pipeline = {
 val assembler = new VectorAssembler().setInputCols((continuousFeatures ).toArray).setOutputCol("features")
 val rf = new RandomForestClassifier().setLabelCol("target").setFeaturesCol("features").
   setMaxBins(500).setNumTrees(100).setSeed(199)
 val pipeline = new Pipeline().setStages(Array(assembler,rf))
 pipeline
}

/// neural network pipeline
def getNeuralPipe( continuousFeatures: Seq[String] ): Pipeline = {
 val assembler = new VectorAssembler().setInputCols((continuousFeatures ).toArray).setOutputCol("features")
 val layers = Array[Int](5, 3, 3, 2)
 val mlp = new MultilayerPerceptronClassifier().
     setLabelCol("target").setFeaturesCol("features").
     setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
 val pipeline = new Pipeline().setStages(Array(assembler,mlp))
 pipeline
}


/// sample and create pipelines
val continuousFeatures = Seq("feature1","feature2","feature3","feature4","feature5")
val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed=199)
val pipe_rf = getRandomPipe( continuousFeatures)
val pipe_mlp = getNeuralPipe( continuousFeatures)
val evaluatorAUROC = new BinaryClassificationEvaluator().setLabelCol("target").setMetricName("areaUnderROC").setRawPredictionCol("probability")


// fit rf model on train and evaluate on test set
val model_rf = pipe_rf.fit(train)
val pred_rf = model_rf.transform(test)
val auc_rf = evaluatorAUROC.evaluate(pred_rf)
model_rf.write.overwrite.save("testrf")


// fit nnet model on train and evaluate on test set
val model_mlp = pipe_mlp.fit(train)
val pred_mlp = model_mlp.transform(test)
val auc_mlp = evaluatorAUROC.evaluate(pred_mlp)
model_mlp.write.overwrite.save("testmlp")


