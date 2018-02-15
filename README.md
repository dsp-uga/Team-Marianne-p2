# Team Marianne-p2: Malware Classification on Spark

This was a course project for CSCI 8360 Data Science Practicum at UGA to implement malware classification on nearly 0.5 TB of data. The aim of the project was to implement everything in RDDs in spark and deploy it to Google Cloud dataproc cluster. Spark's MLLIB was used extensively to implement different algorithms namely SVM, Linear regression, Naive Bayes and Random Forest.

## How to run

A simple way to run the project is implement below command:

```
spark-submit main.py <args>
```
If this repo is cloned as it it, sample data is available in /data directory. If given no arguments for training and testing data path, default path will be used which is in /data directory.

In <args> -model argument indicates which Machine Algorithm will be executed. 
  - rf for Random Forest
  - nbs for Naive Bayes (MLLIB version)
  - lr for Logistic Regression
  - svm for Support Vector Machine (SVM)

## Internal Details

Very simple pipe line for this project can be described as below:
1. Byte file names are stored as keys and extracted byte file is stored as individual value in RDD
2. byte file is tokenized and tf (term frequency) of the bigram is used as feature
3. This RDD is then converted to svmlib file format where source method of MLUTILS was implemented in the project to allow faster     processing of RDDs (Refer to [#1])
4. After converting the data to svmlib format, it is inputted to any of the Machine learning model specified in `<args>` (Refer to [2]).

## References
[1] https://spark.apache.org/docs/1.6.3/api/python/_modules/pyspark/mllib/util.html
[2] https://spark.apache.org/docs/2.2.0/mllib-guide.html#mllib-rdd-based-api

  
