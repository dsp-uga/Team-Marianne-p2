
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel, LogisticRegressionWithLBFGS, LogisticRegressionModel, SVMWithSGD, SVMModel

class SparkRDDMl:
    ''' This class provides functions to be used to implement ML models in rdds
    '''
    def __init__(self, sc) :
        self.sc = sc

    def random_forest_classification(self, sc, args, train_data, test_data):
        '''This does the Random forest classification for given train and test set
        '''
        # Create model and make prediction
        model = RandomForest.trainClassifier(train_data.values(), numClasses=9, categoricalFeaturesInfo={},\
                                         numTrees=100, featureSubsetStrategy="auto",\
                                         impurity='gini', maxDepth=4, maxBins=32)
        test_data.mapValues(lambda x: model.predict(x.features))
    	predictions = model.predict(test_data.values().map(lambda x: x.features))
        if(args.evaluate):
            score(predictions,test_data, model, args.mlModel)
        write_output(predictions, args.output)

    def naive_bayes_mllib(self, sc, args, train_data, test_data):
    	'''This does the Naive Bayes Classification for given train and test set
    	'''
    	# Create model and make predictions
    	model = NaiveBayes.train(train_data, 1.0)
    	predictions = model.predict(test_data.map(lambda x: x.features))
    	if(args.evaluate):
    		score(predictions,test_data, model, args.mlModel)
    	write_output(predictions, args.output)

    def logistic_regression_classification(self, sc, args, train_data, test_data):
    	'''This does the logistic regression classification for given train and test set
    	'''
    	# create model and make prediction
    	model = LogisticRegressionWithLBFGS.train(train_data, iterations=100, numClasses=9)
    	predictions = model.predict(test_data.map(lambda x: x.features))
    	if(args.evaluate):
    		score(predictions,test_data, model, args.mlModel)
    	write_output(predictions, args.output)

    def svm_classification(self, sc, args, train_data, test_data):
    	'''This does the svm classification for given train and test set
    	'''
    	# create model and make prediction
    	model = SVMWithSGD.train(train_data, iterations=100, numClasses=9)
    	predictions = model.predict(test_data.map(lambda x: x.features))
    	if(args.evaluate):
    		score(predictions,test_data, model, args.mlModel)
    	write_output(predictions, args.output)

    def write_output(predictions, path):
    	'''Write the output to file given in the args output path
    	'''
    	resTrain = predictions.collect()
    	f = open(path, 'w')
    	for i in range(len(resTrain)):
    		f.write(str(int(resTrain[i]+1)) + '\n')
    	f.close()
    	return path

    def score(predictions,test_data, model, mlModel):
    	'''Prints out the accuracy accuracy
    	'''
    	print('score called')
    	labelsAndPredictions = test_data.map(lambda x: x.label).zip(predictions)
    	testErr = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(test_data.count())
    	print('For '+mlModel+' Accuracy = ' + str(testErr))
