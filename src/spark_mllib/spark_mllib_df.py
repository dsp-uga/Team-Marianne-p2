from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class SparkDFMl:
    '''This class provides functions to be used to implement ML models in dataframes
    '''
    def __init__(self, sc):
        self.sc = sc

    def featurize_data(sc, data):
        # Tokenize the strings
        tokenizer = Tokenizer(inputCol='feature', outputCol='feature_tokens')
        tokenized_train_df = tokenizer.transform(data)
        #tokenized_train_df.show()
        # Extract bigrams out of strings
        ngram = NGram(n=2, inputCol='feature_tokens', outputCol='ngrams')
        bigram_train_df = ngram.transform(tokenized_train_df)
        # Converting to hashing features
        hashing_tf = HashingTF(inputCol='ngrams', outputCol='rawFeatures')
        tf = hashing_tf.transform(bigram_train_df)
        # Converting to counts to TFIDF
        idf = IDF(inputCol='rawFeatures', outputCol='features')
        idf_model = idf.fit(tf)
        rescaled_data = idf_model.transform(tf)
        string_indexer = StringIndexer(inputCol='label', outputCol='label_numeric')
        rescaled_data_numeric = string_indexer.fit(rescaled_data).transform(rescaled_data)
        ret_df = rescaled_data_numeric.selectExpr('id as id', 'label_numeric as label', 'features as features')
        #print('tf columns are-- ',tf.columns)
        #print('rescaled_data columns are-- ',rescaled_data.columns)
        #print('Joined columns are-- ',tf.join(rescaled_data, ['id']).columns)
        #tf.show()
        #rescaled_data.show()
        #tfidf = tf.join(rescaled_data, ['id', 'label']).select('id', 'label', (tf.rawFeatures*rescaled_data.features_idf).alias('feature'))
        return ret_df

    def naive_bayes(sc, train_df, test_df):
        '''This is implementation of Naive Bayes Algorithm using dataframes
        '''
        train_data = SparkDFMl(sc).featurize_data(train_df)
        test_data = SparkDFMl(sc).featurize_data(test_df)
        nb = NaiveBayes(smoothing=1.0, modelType='multinomial')
        nb_model = nb.fit(train_data)
        predictions = nb_model.transform(test_data)
        predictions.show()

        evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
        accuracy = evaluator.evaluate(prediction)
        print('Accuracy is---' + str(accuracy))
