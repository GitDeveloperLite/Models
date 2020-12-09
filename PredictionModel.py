import sys
import os
os.environ['HADOOP_HOME'] = "home/ec2-user/spark-2.0.0-bin-hadoop2.7"
sys.path.append("home/ec2-user/spark-2.0.0-bin-hadoop2.7/bin")
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import avg
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.ml.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from collections import Counter


def create_df(path):
    spark_read=spark.read.csv(path,header='true',inferSchema='true',sep=';')
    spark_read_pd=spark_read.toPandas()
    spark_read_pd.columns=spark_read_pd.columns.str.strip('""')
    return spark.createDataFrame(spark_read_pd)

#Below prints contents
with open(sys.argv[1], 'r') as f:
   contents = f.read()
#Storing path of the csv file provided
if len(sys.argv) > 1:
    path=sys.argv[1]
#Starting Spark Session
spark = SparkSession.builder.appName("Predict Model").getOrCreate()
#Getting a data frame from the path
file_df=create_df(path)
#Displaying the dataframe created
print("Showing file df before printschema")
print(file_df)
file_df.printSchema()
#Assmebling as vectors for analysis
assembler=VectorAssembler().setInputCols(['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']).setOutputCol('features')
output_data=assembler.transform(file_df)
file_df.show()
print("Showing Output data")
output_data.show()
#Beginning Training  of model
train,test=output_data.randomSplit([0.7,0.3])
model=LogisticRegression(labelCol='quality')
#Fitting the model
model=model.fit(train)
summary=model.summary
summary.predictions.describe().show()
#Getting the predictions
predictions=model.evaluate(test)
print("Showing predictions")
predictions.predictions.show()
predictions.predictions.select('rawPrediction', 'prediction', 'probability').show(10)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
stages=[]
rf = RandomForestClassifier(numTrees=100, maxDepth=5, maxBins=5, labelCol="features",featuresCol="features",seed=42)
stages += [rf]
#Creating a pipeline
pipeline = Pipeline(stages = stages)
lr = LogisticRegression().setFeaturesCol("features")
params=ParamGridBuilder().build()
#Populating CrossValidator with the 
cv = CrossValidator(estimator=pipeline,
            estimatorParamMaps=params,
            evaluator=evaluator,
            numFolds=5)
cvModel=cv.fit(output_data.select('features')) 
predictions=cvModel.transform(test)
predictions_pandas=predictions.toPandas()
#Obtaining the f1 score
f1 = f1_score(predictions_pandas.lable, predictions_pandas.prediction, average='weighted')
print(f1)
