# SparkFileStreaming
Read a ML model from file, then watch a folder for new batches to be scored

### 1. Built a Spark ML pipeline random forest model and tested the loading & scoring with the *.scala files
### 2. The RF model is stored in "testrf" folder
### 3. Change the file locations and spark version as needed in build.sbt
### 4. build the project with sbt package
### 5. Run the project: spark-submit --class com.mapr.streamtest.Main target/scala-2.11/filestreaming_2.11-0.1-SNAPSHOT.jar
### 6. In another window, copy the file sample.csv into the stream_test folder
