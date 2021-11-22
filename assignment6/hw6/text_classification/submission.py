"""
Text classification
"""

import util
import operator
import string
from collections import Counter
from collections import defaultdict

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        if (k<0):
            self.blacklist = set(blacklist)
        else:
            self.blacklist = set(blacklist[0:k])
        self.n = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        count = 0
        for bad_word in self.blacklist:
            if bad_word in text:
                count = count + 1
                if count >= self.n:
                    return -1
        return 1
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    word_vector = {}
    words = x.split()
    for word in words:
        if word in word_vector:
            word_vector[word] += 1
        else:
            word_vector[word] = 1

    return word_vector



class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params
        self.resmap = {}

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        result_map = self.featureFunction(x)
        self.resmap = result_map
        val = 0
        for key in result_map:
            if key in self.params:
                val += result_map[key] * self.params[key]

        return val

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # return defaultdict(float)
    w = {}
    for iter in range(iters):
        for text, label in trainExamples:
            y = 0
            if label == labels[0]:
                y = 1
            else:
                y = -1

            classifier = WeightedClassifier(labels, featureExtractor, w)
            val = 1 if classifier.classify(text) >= 0 else -1
            if val != y:
                result_map = classifier.resmap

                for key in result_map:
                    if key not in w:
                        w[key] = 0

                for key in result_map:
                    w[key] = w[key] + result_map[key] * y
    return w



def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    result_map = extractUnigramFeatures(x)
    words = x.split()
    for i in range(len(words)):
        bigram = ""
        if i == 0 or words[i - 1] in string.punctuation:
            bigram = "-BEGIN- " + words[i]
        else:
            bigram = words[i - 1] + " " + words[i]

        if bigram in result_map:
            result_map[bigram] += 1
        else:
            result_map[bigram] = 1

    return result_map


class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        raise NotImplementedError("TODO:")       
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        raise NotImplementedError("TODO:")       
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        raise NotImplementedError("TODO:")       
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    raise NotImplementedError("TODO:")       
    # END_YOUR_CODE

