# Team Marianne-p2: Malware Classification on Spark

This was a course project for CSCI 8360 Data Science Practicum at UGA to implement malware classification on nearly 0.5 TB of data. The aim of the project was to implement everything in RDDs in spark and deploy it to Google Cloud dataproc cluster. Spark's MLLIB was used extensively to implement different algorithms namely SVM, Linear regression, Naive Bayes and Random Forest.

## How to run

A simple way to run the project is implement below command:

```
spark-submit main.py <args>
```
If this repo is cloned as it it, sample data is available in /data directory. If given no arguments for training and testing data path, default path will be used which is in /data directory.

Complete list of args:
 - "-d", "--dataset" for Path to text file containing hash of documents in training set
 
 - "-l", "--labels" for Path to text file containing labels of documents in training set
 
 - "-t", "--testset" for Path to text file containing hash of documents in testing set

 - "-e", "--evaluate" for Set this to evaluate accuracy on the test set

 - "m", "--testlabels" for Path to text file containing labels of documents in testing set. If evaluate is set true, this file is compared with classifier output

 - "-a", "--asmtrain" for Path to directory that contains asm documemts of training set

 - "-at", "--asmtest" for Path to directory that contains asm documemts of testing set

 - "-b", "--bytestrain" for Path to directory that contains bytes documemts of training set

 - "-bt", "--bytestest" for Path to directory that contains bytes documemts of testing set

 - "-A", "--asmrdd" for Path to text file in which RDD from asm file is stored after preprocessing

 - "-B", "--bytesrdd" for Path to text file in which RDD from bytes file is stored after preprocessing

 - "-C", "--bytesrddTest" for Path to text file in which RDD from bytes file is stored after preprocessing for test

 - "-o", "--output" for Path to the directory where output will be written

 - "-model", "--mlModel" for Specify which ML model is to be used

The options for `-model` argument that indicates which ML Algorithm will be executed are as follows 
  - "rf" for Random Forest
  - "nbs" for Naive Bayes (MLLIB version)
  - "lr" for Logistic Regression
  - "svm" for Support Vector Machine (SVM)
  - "nbs_df" for Naive Bayes based on Dataframes

## Internal Details

Very simple pipe line for this project can be described as below:
1. Byte file names are stored as keys and extracted byte file is stored as individual value in RDD
2. byte file is tokenized and tf (term frequency) of the bigram is used as feature
3. This RDD is then converted to svmlib file format where source method of MLUTILS was implemented in the project to allow faster     processing of RDDs (Refer to [#1])
4. After converting the data to svmlib format, it is inputted to any of the Machine learning model specified in `<args>` (Refer to [2]).

Another Dataframe based implementation also has very simple pipeline:
1. Byte file names are stored as keys and extracted byte file is stored as individual value in RDD
2. byte file is tokenized and tfidf value is extracted from the bigrams features
3. Naive bayes algorithm is then applied on the tfidf features to predict the labels

Dataframe based implementation currently supports only Naive Bayes algorithm but easily, other models (supported by spark MLLIB) can be added as the data preparation module is already completed

## Results

We got nearly average of 80 % accuracy on small dataset by implementing all the implementations of ML models but we majorly struggled with running the model on large dataset. The reason might be that the code is not scaling up.

## References
[1] https://spark.apache.org/docs/1.6.3/api/python/_modules/pyspark/mllib/util.html 

[2] https://spark.apache.org/docs/2.2.0/mllib-guide.html#mllib-rdd-based-api

  
