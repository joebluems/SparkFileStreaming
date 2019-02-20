# SparkFileStreaming
Loads ML models from files, then watches the folder "stream_test" for new batches to be scored

### 1. Pre-built models - code for training and testing is in the scala files...
### 2. Location of models used by streaming - ./testrf, ./testmlp and ./testkmeans
### 3. Change the file locations and spark version as needed in build.sbt
### 4. build the project: > sbt package
### 5. Run the project:  > sbt run
### 6. Send a batch in another window: > head -10 sample1000.csv > stream_test/1.csv

### Note: currently the model calculates scores for all models but only outputs one set.
