# Databricks notebook source exported at Fri, 8 Jul 2016 15:59:46 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC ![ML Logo](http://spark-mooc.github.io/web-assets/images/CS190.1x_Banner_300.png)
# MAGIC # **Click-Through Rate Prediction Lab**
# MAGIC This lab covers the steps for creating a click-through rate (CTR) prediction pipeline.  You will work with the [Criteo Labs](http://labs.criteo.com/) dataset that was used for a recent [Kaggle competition](https://www.kaggle.com/c/criteo-display-ad-challenge).
# MAGIC 
# MAGIC ** This lab will cover: **
# MAGIC 
# MAGIC * *Part 1:* Featurize categorical data using one-hot-encoding (OHE)
# MAGIC 
# MAGIC * *Part 2:* Construct an OHE dictionary
# MAGIC 
# MAGIC * *Part 3:* Parse CTR data and generate OHE features
# MAGIC  * *Visualization 1:* Feature frequency
# MAGIC 
# MAGIC * *Part 4:* CTR prediction and logloss evaluation
# MAGIC  * *Visualization 2:* ROC curve
# MAGIC 
# MAGIC * *Part 5:* Reduce feature dimension via feature hashing
# MAGIC 
# MAGIC > Note that, for reference, you can look up the details of:
# MAGIC > * the relevant Spark methods in [PySpark's DataFrame API](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.DataFrame)
# MAGIC > * the relevant NumPy methods in the [NumPy Reference](http://docs.scipy.org/doc/numpy/reference/index.html)

# COMMAND ----------

labVersion = 'cs120x-criteo-1.0.0'

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 1: Featurize categorical data using one-hot-encoding **

# COMMAND ----------

# MAGIC %md
# MAGIC ** (1a) One-hot-encoding **
# MAGIC 
# MAGIC We would like to develop code to convert categorical features to numerical ones, and to build intuition, we will work with a sample unlabeled dataset with three data points, with each data point representing an animal. The first feature indicates the type of animal (bear, cat, mouse); the second feature describes the animal's color (black, tabby); and the third (optional) feature describes what the animal eats (mouse, salmon).
# MAGIC 
# MAGIC In a one-hot-encoding (OHE) scheme, we want to represent each tuple of `(featureID, category)` via its own binary feature.  We can do this in Python by creating a dictionary that maps each tuple to a distinct integer, where the integer corresponds to a binary feature. To start, manually enter the entries in the OHE dictionary associated with the sample dataset by mapping the tuples to consecutive integers starting from zero,  ordering the tuples first by featureID and next by category.
# MAGIC 
# MAGIC Later in this lab, we'll use OHE dictionaries to transform data points into compact lists of features that can be used in machine learning algorithms.

# COMMAND ----------

sqlContext.setConf('spark.sql.shuffle.partitions', '8')  # Set default partitions for DataFrame operations

# COMMAND ----------

from collections import defaultdict
# Data for manual OHE
# Note: the first data point does not include any value for the optional third feature
sampleOne = [(0, 'mouse'), (1, 'black')]
sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]

def sampleToRow(sample):
    tmpDict = defaultdict(lambda: None)
    tmpDict.update(sample)
    return [tmpDict[i] for i in range(3)]

sqlContext.createDataFrame(map(sampleToRow, [sampleOne, sampleTwo, sampleThree]),
                           ['animal', 'color', 'food']).show()

sampleDataDF = sqlContext.createDataFrame([(sampleOne,), (sampleTwo,), (sampleThree,)], ['features'])
sampleDataDF.show(truncate=False)

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
sampleOHEDictManual = {}
sampleOHEDictManual[(0,'bear')] = <FILL IN>
sampleOHEDictManual[(0,'cat')] = <FILL IN>
sampleOHEDictManual[(0,'mouse')] = <FILL IN>
sampleOHEDictManual<FILL IN>
sampleOHEDictManual<FILL IN>
sampleOHEDictManual<FILL IN>
sampleOHEDictManual<FILL IN>

# COMMAND ----------

# TEST One-hot-encoding (1a)
from test_helper import Test

Test.assertEqualsHashed(sampleOHEDictManual[(0,'bear')],
                        'b6589fc6ab0dc82cf12099d1c2d40ab994e8410c',
                        "incorrect value for sampleOHEDictManual[(0,'bear')]")
Test.assertEqualsHashed(sampleOHEDictManual[(0,'cat')],
                        '356a192b7913b04c54574d18c28d46e6395428ab',
                        "incorrect value for sampleOHEDictManual[(0,'cat')]")
Test.assertEqualsHashed(sampleOHEDictManual[(0,'mouse')],
                        'da4b9237bacccdf19c0760cab7aec4a8359010b0',
                        "incorrect value for sampleOHEDictManual[(0,'mouse')]")
Test.assertEqualsHashed(sampleOHEDictManual[(1,'black')],
                        '77de68daecd823babbb58edb1c8e14d7106e83bb',
                        "incorrect value for sampleOHEDictManual[(1,'black')]")
Test.assertEqualsHashed(sampleOHEDictManual[(1,'tabby')],
                        '1b6453892473a467d07372d45eb05abc2031647a',
                        "incorrect value for sampleOHEDictManual[(1,'tabby')]")
Test.assertEqualsHashed(sampleOHEDictManual[(2,'mouse')],
                        'ac3478d69a3c81fa62e60f5c3696165a4e5e6ac4',
                        "incorrect value for sampleOHEDictManual[(2,'mouse')]")
Test.assertEqualsHashed(sampleOHEDictManual[(2,'salmon')],
                        'c1dfd96eea8cc2b62785275bca38ac261256e278',
                        "incorrect value for sampleOHEDictManual[(2,'salmon')]")
Test.assertEquals(len(sampleOHEDictManual.keys()), 7,
                  'incorrect number of keys in sampleOHEDictManual')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (1b) Sparse vectors **
# MAGIC 
# MAGIC Data points can typically be represented with a small number of non-zero OHE features relative to the total number of features that occur in the dataset.  By leveraging this sparsity and using sparse vector representations of OHE data, we can reduce storage and computational burdens.  Below are a few sample vectors represented as dense numpy arrays.  Use [SparseVector](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.SparseVector) to represent them in a sparse fashion, and verify that both the sparse and dense representations yield the same results when computing [dot products](http://en.wikipedia.org/wiki/Dot_product) (we will later use MLlib to train classifiers via gradient descent, and MLlib will need to compute dot products between SparseVectors and dense parameter vectors).
# MAGIC 
# MAGIC Use `SparseVector(size, *args)` to create a new sparse vector where size is the length of the vector and args is either:
# MAGIC 1. A list of indices and a list of values corresponding to the indices. The indices list must be sorted in ascending order. For example, SparseVector(5, [1, 3, 4], [10, 30, 40]) will represent the vector [0, 10, 0, 30, 40]. The non-zero indices are 1, 3 and 4. On the other hand, SparseVector(3, [2, 1], [5, 5]) will give you an error because the indices list [2, 1] is not in ascending order. Note: you cannot simply sort the indices list, because otherwise the values will not correspond to the respective indices anymore.
# MAGIC 2. A list of (index, value) pair. In this case, the indices need not be sorted. For example, SparseVector(5, [(3, 1), (1, 2)]) will give you the vector [0, 2, 0, 1, 0].
# MAGIC 
# MAGIC SparseVectors are much more efficient when working with sparse data because they do not store zero values (only store non-zero values and their indices). You'll need to create a sparse vector representation of each dense vector `aDense` and `bDense`.

# COMMAND ----------

import numpy as np
from pyspark.mllib.linalg import SparseVector

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
aDense = np.array([0., 3., 0., 4.])
aSparse = <FILL IN>

bDense = np.array([0., 0., 0., 1.])
bSparse = <FILL IN>

w = np.array([0.4, 3.1, -1.4, -.5])
print aDense.dot(w)
print aSparse.dot(w)
print bDense.dot(w)
print bSparse.dot(w)

# COMMAND ----------

# TEST Sparse Vectors (1b)
Test.assertTrue(isinstance(aSparse, SparseVector), 'aSparse needs to be an instance of SparseVector')
Test.assertTrue(isinstance(bSparse, SparseVector), 'aSparse needs to be an instance of SparseVector')
Test.assertTrue(aDense.dot(w) == aSparse.dot(w),
                'dot product of aDense and w should equal dot product of aSparse and w')
Test.assertTrue(bDense.dot(w) == bSparse.dot(w),
                'dot product of bDense and w should equal dot product of bSparse and w')
Test.assertTrue(aSparse.numNonzeros() == 2, 'aSparse should not store zero values')
Test.assertTrue(bSparse.numNonzeros() == 1, 'bSparse should not store zero values')

# COMMAND ----------

# MAGIC %md
# MAGIC **(1c) OHE features as sparse vectors **
# MAGIC 
# MAGIC Now let's see how we can represent the OHE features for points in our sample dataset.  Using the mapping defined by the OHE dictionary from Part (1a), manually define OHE features for the three sample data points using SparseVector format.  Any feature that occurs in a point should have the value 1.0.  For example, the `DenseVector` for a point with features 2 and 4 would be `[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]`.

# COMMAND ----------

# Reminder of the sample features
# sampleOne = [(0, 'mouse'), (1, 'black')]
# sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
# sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
sampleOneOHEFeatManual = <FILL IN>
sampleTwoOHEFeatManual = <FILL IN>
sampleThreeOHEFeatManual = <FILL IN>

# COMMAND ----------

# TEST OHE Features as sparse vectors (1c)
Test.assertTrue(isinstance(sampleOneOHEFeatManual, SparseVector),
                'sampleOneOHEFeatManual needs to be a SparseVector')
Test.assertTrue(isinstance(sampleTwoOHEFeatManual, SparseVector),
                'sampleTwoOHEFeatManual needs to be a SparseVector')
Test.assertTrue(isinstance(sampleThreeOHEFeatManual, SparseVector),
                'sampleThreeOHEFeatManual needs to be a SparseVector')
Test.assertEqualsHashed(sampleOneOHEFeatManual,
                        'ecc00223d141b7bd0913d52377cee2cf5783abd6',
                        'incorrect value for sampleOneOHEFeatManual')
Test.assertEqualsHashed(sampleTwoOHEFeatManual,
                        '26b023f4109e3b8ab32241938e2e9b9e9d62720a',
                        'incorrect value for sampleTwoOHEFeatManual')
Test.assertEqualsHashed(sampleThreeOHEFeatManual,
                        'c04134fd603ae115395b29dcabe9d0c66fbdc8a7',
                        'incorrect value for sampleThreeOHEFeatManual')

# COMMAND ----------

# MAGIC %md
# MAGIC **(1d) Define a OHE function **
# MAGIC 
# MAGIC Next we will use the OHE dictionary from Part (1a) to programatically generate OHE features from the original categorical data.  First write a function called `oneHotEncoding` that creates OHE feature vectors in `SparseVector` format.  Then use this function to create OHE features for the first sample data point and verify that the result matches the result from Part (1c).
# MAGIC 
# MAGIC > Note: We'll pass the OHE dictionary in as a [Broadcast](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.Broadcast) variable, which will greatly improve performance when we call this function as part of a UDF.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def oneHotEncoding(rawFeats, oheDictBroadcast, numOHEFeats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        You should ensure that the indices used to create a SparseVector are sorted.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        oheDictBroadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indices equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """
    <FILL IN>

# Calculate the number of features in sampleOHEDictManual
numSampleOHEFeats = <FILL IN>
sampleOHEDictManualBroadcast = sc.broadcast(sampleOHEDictManual)

# Run oneHotEnoding on sampleOne.  Make sure to pass in the Broadcast variable.
sampleOneOHEFeat = <FILL IN>

print sampleOneOHEFeat

# COMMAND ----------

# TEST Define an OHE Function (1d)
Test.assertTrue(sampleOneOHEFeat == sampleOneOHEFeatManual,
                'sampleOneOHEFeat should equal sampleOneOHEFeatManual')
Test.assertEquals(sampleOneOHEFeat, SparseVector(7, [2,3], [1.0,1.0]),
                  'incorrect value for sampleOneOHEFeat')
Test.assertEquals(oneHotEncoding([(1, 'black'), (0, 'mouse')], sampleOHEDictManualBroadcast,
                                 numSampleOHEFeats), SparseVector(7, [2,3], [1.0,1.0]),
                  'incorrect definition for oneHotEncoding')

# COMMAND ----------

# MAGIC %md
# MAGIC **(1e) Apply OHE to a dataset **
# MAGIC 
# MAGIC Finally, use the function from Part (1d) to create OHE features for all 3 data points in the sample dataset.  You'll need to generate a [UDF](https://spark.apache.org/docs/1.6.1/api/python/pyspark.sql.html#pyspark.sql.functions.udf) that can be used in a `DataFrame` `select` statement.
# MAGIC 
# MAGIC > Note: Your implemenation of `oheUDFGenerator` needs to call your `oneHotEncoding` function.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import VectorUDT

def oheUDFGenerator(oheDictBroadcast):
    """Generate a UDF that is setup to one-hot-encode rows with the given dictionary.

    Note:
        We'll reuse this function to generate a UDF that can one-hot-encode rows based on a
        one-hot-encoding dictionary built from the training data.  Also, you should calculate
        the number of features before calling the oneHotEncoding function.

    Args:
        oheDictBroadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.

    Returns:
        UserDefinedFunction: A UDF can be used in `DataFrame` `select` statement to call a
            function on each row in a given column.  This UDF should call the oneHotEncoding
            function with the appropriate parameters.
    """
    length = <FILL IN>
    return udf(lambda x: <FILL IN>, VectorUDT())

sampleOHEDictUDF = oheUDFGenerator(sampleOHEDictManualBroadcast)
sampleOHEDF = sampleDataDF.select(<FILL IN>)
sampleOHEDF.show(truncate=False)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.mllib.linalg import VectorUDT

def oheUDFGenerator(oheDictBroadcast):
    """Generate a UDF that is setup to one-hot-encode rows with the given dictionary.

    Note:
        We'll reuse this function to generate a UDF that can one-hot-encode rows based on a
        one-hot-encoding dictionary built from the training data.  Also, you should calculate
        the number of features before calling the oneHotEncoding function.

    Args:
        oheDictBroadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.

    Returns:
        UserDefinedFunction: A UDF can be used in `DataFrame` `select` statement to call a
            function on each row in a given column.  This UDF should call the oneHotEncoding
            function with the appropriate parameters.
    """
    length = len(oheDictBroadcast.value.keys())
    return udf(lambda x: oneHotEncoding(x, oheDictBroadcast, length), VectorUDT())

sampleOHEDictUDF = oheUDFGenerator(sampleOHEDictManualBroadcast)
sampleOHEDF = sampleDataDF.select(sampleOHEDictUDF('features'))
sampleOHEDF.show(truncate=False)

# COMMAND ----------

# TEST Apply OHE to a dataset (1e)
sampleOHEDataValues = sampleOHEDF.collect()
Test.assertTrue(len(sampleOHEDataValues) == 3, 'sampleOHEData should have three elements')
Test.assertEquals(sampleOHEDataValues[0], (SparseVector(7, {2: 1.0, 3: 1.0}),),
                  'incorrect OHE for first sample')
Test.assertEquals(sampleOHEDataValues[1], (SparseVector(7, {1: 1.0, 4: 1.0, 5: 1.0}),),
                  'incorrect OHE for second sample')
Test.assertEquals(sampleOHEDataValues[2], (SparseVector(7, {0: 1.0, 3: 1.0, 6: 1.0}),),
                  'incorrect OHE for third sample')
Test.assertTrue('oneHotEncoding' in sampleOHEDictUDF.func.func_code.co_names,
                'oheUDFGenerator should call oneHotEncoding')

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 2: Construct an OHE dictionary **

# COMMAND ----------

# MAGIC %md
# MAGIC **(2a) DataFrame with rows of `(featureID, category)` **
# MAGIC 
# MAGIC To start, create a DataFrame of distinct `(featureID, category)` tuples. In our sample dataset, the 7 items in the resulting DataFrame are `(0, 'bear')`, `(0, 'cat')`, `(0, 'mouse')`, `(1, 'black')`, `(1, 'tabby')`, `(2, 'mouse')`, `(2, 'salmon')`. Notably `'black'` appears twice in the dataset but only contributes one item to the DataFrame: `(1, 'black')`, while `'mouse'` also appears twice and contributes two items: `(0, 'mouse')` and `(2, 'mouse')`.  Use [explode](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.explode) and [distinct](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.distinct).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import explode
sampleDistinctFeatsDF = (sampleDataDF
                         <FILL IN>)
sampleDistinctFeatsDF.show()

# COMMAND ----------

# TEST Pair RDD of (featureID, category) (2a)
Test.assertEquals(sorted(map(lambda r: r[0], sampleDistinctFeatsDF.collect())),
                  [(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'),
                   (1, 'tabby'), (2, 'mouse'), (2, 'salmon')],
                  'incorrect value for sampleDistinctFeats')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (2b) OHE Dictionary from distinct features **
# MAGIC 
# MAGIC Next, create an RDD of key-value tuples, where each `(featureID, category)` tuple in `sampleDistinctFeatsDF` is a key and the values are distinct integers ranging from 0 to (number of keys - 1).  Then convert this RDD into a dictionary, which can be done using the `collectAsMap` action.  Note that there is no unique mapping from keys to values, as all we require is that each `(featureID, category)` key be mapped to a unique integer between 0 and the number of keys.  In this exercise, any valid mapping is acceptable.  Use [zipWithIndex](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.zipWithIndex) followed by [collectAsMap](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.collectAsMap).
# MAGIC 
# MAGIC In our sample dataset, one valid list of key-value tuples is: `[((0, 'bear'), 0), ((2, 'salmon'), 1), ((1, 'tabby'), 2), ((2, 'mouse'), 3), ((0, 'mouse'), 4), ((0, 'cat'), 5), ((1, 'black'), 6)]`. The dictionary defined in Part (1a) illustrates another valid mapping between keys and integers.
# MAGIC 
# MAGIC > Note: We provide the code to convert the DataFrame to an RDD.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
sampleOHEDict = (sampleDistinctFeatsDF
                 .rdd
                 .map(lambda r: tuple(r[0]))
                 <FILL IN>)
print sampleOHEDict

# COMMAND ----------

# TEST OHE Dictionary from distinct features (2b)
Test.assertEquals(sorted(sampleOHEDict.keys()),
                  [(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'),
                   (1, 'tabby'), (2, 'mouse'), (2, 'salmon')],
                  'sampleOHEDict has unexpected keys')
Test.assertEquals(sorted(sampleOHEDict.values()), range(7), 'sampleOHEDict has unexpected values')

# COMMAND ----------

# MAGIC %md
# MAGIC **(2c) Automated creation of an OHE dictionary **
# MAGIC 
# MAGIC Now use the code from Parts (2a) and (2b) to write a function that takes an input dataset and outputs an OHE dictionary.  Then use this function to create an OHE dictionary for the sample dataset, and verify that it matches the dictionary from Part (2b).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def createOneHotDict(inputDF):
    """Creates a one-hot-encoder dictionary based on the input data.

    Args:
        inputDF (DataFrame with 'feature' column): A DataFrame where each row contains a list of
            (featureID, value) tuples.

    Returns:
        dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
            unique integers.
    """
    <FILL IN>

sampleOHEDictAuto = createOneHotDict(sampleDataDF)

# COMMAND ----------

# TEST Automated creation of an OHE dictionary (2c)
Test.assertEquals(sorted(sampleOHEDictAuto.keys()),
                  [(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'),
                   (1, 'tabby'), (2, 'mouse'), (2, 'salmon')],
                  'sampleOHEDictAuto has unexpected keys')
Test.assertEquals(sorted(sampleOHEDictAuto.values()), range(7),
                  'sampleOHEDictAuto has unexpected values')

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Part 3: Parse CTR data and generate OHE features**

# COMMAND ----------

# MAGIC %md
# MAGIC Before we can proceed, you'll first need to obtain the data from Criteo.  Here is the link to Criteo's data sharing agreement: [http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/](http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/).  After you accept the agreement, you can obtain the download URL by right-clicking on the "Download Sample" button and clicking "Copy link address" or "Copy Link Location", depending on your browser.  Paste the URL into the `# TODO` cell below. The script below will download the file and make the sample dataset's contents available in the `rawData` variable.
# MAGIC 
# MAGIC Note that the download should complete within 30 seconds.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
import glob
from io import BytesIO
import os.path
import tarfile
import urllib2
import urlparse

# Paste in url, url should end with: dac_sample.tar.gz
url = '<FILL IN>'

url = url.strip()

if 'rawDF' in locals():
    print 'rawDF already loaded.  Nothing to do.'
elif not url.endswith('dac_sample.tar.gz'):
    print 'Check your download url.  Are you downloading the Sample dataset?'
else:
    try:
        tmp = BytesIO()

        hdr = { 'User-Agent': 'Databricks' }

        req = urllib2.Request(url, headers=hdr)

        urlHandle = urllib2.urlopen(req)
        tmp.write(urlHandle.read())
        tmp.seek(0)

        tarFile = tarfile.open(fileobj=tmp)

        dacSample = tarFile.extractfile('dac_sample.txt')
        dacSample = [unicode(x.replace('\n', '').replace('\t', ',')) for x in dacSample]
        rawDF  = sqlContext.createDataFrame(sc.parallelize(map(lambda x: (x,), dacSample), 8), ['text'])

        print 'rawDF loaded from url'
        print rawDF.take(1)
    except IOError:
        print 'Unable to unpack: {0}'.format(url)

# COMMAND ----------

# MAGIC %md
# MAGIC **(3a) Loading and splitting the data **
# MAGIC 
# MAGIC We are now ready to start working with the actual CTR data, and our first task involves splitting it into training, validation, and test sets.  Use the [randomSplit method](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit) with the specified weights and seed to create DFs storing each of these datasets, and then [cache](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.cache) each of these DFs, as we will be accessing them multiple times in the remainder of this lab. Finally, compute the size of each dataset.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
weights = [.8, .1, .1]
seed = 42

# Use randomSplit with weights and seed
rawTrainDF, rawValidationDF, rawTestDF = rawDF.<FILL IN>

# Cache and count the DataFrames
nTrain = rawTrainDF.<FILL IN>
nVal = rawValidationDF.<FILL IN>
nTest = rawTestDF.<FILL IN>
print nTrain, nVal, nTest, nTrain + nVal + nTest
rawDF.show(1)

# COMMAND ----------

# TEST Loading and splitting the data (3a)
Test.assertTrue(all([rawTrainDF.is_cached, rawValidationDF.is_cached, rawTestDF.is_cached]),
                'you must cache the split data')
Test.assertEquals(nTrain, 80018, 'incorrect value for nTrain')
Test.assertEquals(nVal, 9966, 'incorrect value for nVal')
Test.assertEquals(nTest, 10016, 'incorrect value for nTest')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3b) Extract features **
# MAGIC 
# MAGIC We will now parse the raw training data to create a DataFrame that we can subsequently use to create an OHE dictionary. Note from the `show()` command in Part (3a) that each raw data point is a string containing several fields separated by some delimiter.  For now, we will ignore the first field (which is the 0-1 label), and parse the remaining fields (or raw features).  To do this, complete the implemention of the `parsePoint` function.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def parsePoint(point):
    """Converts a comma separated string into a list of (featureID, value) tuples.

    Note:
        featureIDs should start at 0 and increase to the number of features - 1.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.

    Returns:
        list: A list of (featureID, value) tuples.
    """
    <FILL IN>

print parsePoint(rawDF.select('text').first()[0])

# COMMAND ----------

# TEST Extract features (3b)
Test.assertEquals(parsePoint(rawDF.select('text').first()[0])[:3], [(0, u'1'), (1, u'1'), (2, u'5')],
                  'incorrect implementation of parsePoint')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3c) Extracting features continued**
# MAGIC 
# MAGIC Next, we'll create a `parseRawDF` function that creates a 'label' column from the first value in the text and a 'tuples' column from the rest of the values.  The 'tuples' column will be created using `parsePointUDF`, which we've provided and is based on your `parsePoint` function.  Note that to name your columns you should use [alias](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.alias).  You can split the 'text' field in `rawDF` using [split](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.split) and retrieve the first value of the resulting array with [getItem](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.getItem).  Your `parseRawDF` function should also cache the DataFrame it returns.

# COMMAND ----------

# TODO
from pyspark.sql.functions import udf, split
from pyspark.sql.types import ArrayType, StructType, StructField, LongType, StringType

parsePointUDF = udf(parsePoint, ArrayType(StructType([StructField('_1', LongType()),
                                                      StructField('_2', StringType())])))

def parseRawDF(rawDF):
    """Convert a DataFrame consisting of rows of comma separated text into labels and tuples.

    Args:
        rawDF (DataFrame with a 'text' column): DataFrame containg the raw comma separated data.

    Returns:
        DataFrame: A DataFrame with 'label' and 'tuple' columns.
    """
    <FILL IN>

# Parse the raw training DataFrame
parsedTrainDF = <FILL IN>

from pyspark.sql.functions import (explode, col)
numCategories = (parsedTrainDF
                 .select(explode('tuples').alias('tuple'))
                 .distinct()
                 .select(col('tuple').getField('_1').alias('featureNumber'))
                 .groupBy('featureNumber')
                 .sum()
                 .orderBy('featureNumber')
                 .collect())

print numCategories[2][1]

# COMMAND ----------

# TEST Extract features (3b)
Test.assertTrue(parsedTrainDF.is_cached, 'parseRawDF should return a cached DataFrame')
Test.assertEquals(numCategories[2][1], 1694, 'incorrect implementation of parsePoint or parseRawDF')
Test.assertEquals(numCategories[32][1], 128, 'incorrect implementation of parsePoint or parseRawDF')

# COMMAND ----------

# MAGIC %md
# MAGIC **(3d) Create an OHE dictionary from the dataset **
# MAGIC 
# MAGIC Note that parsePoint returns a data point as a list of `(featureID, category)` tuples, which is the same format as the sample dataset studied in Parts 1 and 2 of this lab.  Using this observation, create an OHE dictionary from the parsed training data using the function implemented in Part (2c). Note that we will assume for simplicity that all features in our CTR dataset are categorical.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
ctrOHEDict = <FILL IN>
numCtrOHEFeats = <FILL IN>
print numCtrOHEFeats
print ctrOHEDict[(0, '')]

# COMMAND ----------

# TEST Create an OHE dictionary from the dataset (3d)
Test.assertEquals(numCtrOHEFeats, 233940, 'incorrect number of features in ctrOHEDict')
Test.assertTrue((0, '') in ctrOHEDict, 'incorrect features in ctrOHEDict')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3e) Apply OHE to the dataset **
# MAGIC 
# MAGIC Now let's use this OHE dictionary, by starting with the training data that we've parsed into 'label' and 'tuples' columns, to create one-hot-encoded features.  Recall that we created a function `oheUDFGenerator` that can create the UDF that we need to convert 'tuples' into 'features'.  Make sure that `oheTrainDF` contains a 'label' and 'features' column and is cached.

# COMMAND ----------

# TODO
oheDictBroadcast = <FILL IN>
oheDictUDF = <FILL IN>
oheTrainDF = (parsedTrainDF
              <FILL IN>)

print oheTrainDF.count()
print oheTrainDF.take(1)

# COMMAND ----------

# TEST Apply OHE to the dataset (3e)
Test.assertTrue('label' in oheTrainDF.columns and 'features' in oheTrainDF.columns, 'oheTrainDF should have label and features columns')
Test.assertTrue(oheTrainDF.is_cached, 'oheTrainDF should be cached')
numNZ = sum(parsedTrainDF.rdd.map(lambda r: len(r[1])).take(5))
numNZAlt = sum(oheTrainDF.rdd.map(lambda r: len(r[1].indices)).take(5))
Test.assertEquals(numNZ, numNZAlt, 'incorrect value for oheTrainDF')

# COMMAND ----------

# MAGIC %md
# MAGIC **Visualization 1: Feature frequency **
# MAGIC 
# MAGIC We will now visualize the number of times each of the 233,941 OHE features appears in the training data. We first compute the number of times each feature appears, then bucket the features by these counts.  The buckets are sized by powers of 2, so the first bucket corresponds to features that appear exactly once ( \\( \scriptsize 2^0 \\) ), the second to features that appear twice ( \\( \scriptsize 2^1 \\) ), the third to features that occur between three and four ( \\( \scriptsize 2^2 \\) ) times, the fifth bucket is five to eight ( \\( \scriptsize 2^3 \\) ) times and so on. The scatter plot below shows the logarithm of the bucket thresholds versus the logarithm of the number of features that have counts that fall in the buckets.

# COMMAND ----------

from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import log

getIndices = udf(lambda sv: map(int, sv.indices), ArrayType(IntegerType()))
featureCounts = (oheTrainDF
                 .select(explode(getIndices('features')))
                 .groupBy('col')
                 .count()
                 .withColumn('bucket', log('count').cast('int'))
                 .groupBy('bucket')
                 .count()
                 .orderBy('bucket')
                 .collect())

# COMMAND ----------

import matplotlib.pyplot as plt

x, y = zip(*featureCounts)
x, y = x, np.log(y)

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 12, 1), np.arange(0, 14, 2))
ax.set_xlabel(r'$\log_e(bucketSize)$'), ax.set_ylabel(r'$\log_e(countInBucket)$')
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
display(fig)
pass

# COMMAND ----------

# MAGIC %md
# MAGIC **(3f) Handling unseen features **
# MAGIC 
# MAGIC We naturally would like to repeat the process from Part (3e), e.g., to compute OHE features for the validation and test datasets.  However, we must be careful, as some categorical values will likely appear in new data that did not exist in the training data. To deal with this situation, update the `oneHotEncoding()` function from Part (1d) to ignore previously unseen categories, and then compute OHE features for the validation data.  Rember that you can parse a raw DataFrame using `parseRawDF`.
# MAGIC > Note: you'll have to generate a new UDF using `oheUDFGenerator` so that the updated `oneHotEncoding` function is used.  And make sure to cache `oheValidationDF`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def oneHotEncoding(rawFeats, oheDictBroadcast, numOHEFeats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        You should ensure that the indices used to create a SparseVector are sorted, and that the
        function handles missing features.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        oheDictBroadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indices equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """
    <FILL IN>

oheDictMissingUDF = <FILL IN>
oheValidationDF = (<FILL IN>)

oheValidationDF.count()
oheValidationDF.show(1, truncate=False)

# COMMAND ----------

# TEST Handling unseen features (3f)
from pyspark.sql.functions import size, sum as sqlsum

Test.assertTrue(oheValidationDF.is_cached, 'you need to cache oheValidationDF')
numNZVal = (oheValidationDF
            .select(sqlsum(size(getIndices('features'))))
            .first()[0])

Test.assertEquals(numNZVal, 368027, 'incorrect number of features')

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 4: CTR prediction and logloss evaluation **

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4a) Logistic regression **
# MAGIC 
# MAGIC We are now ready to train our first CTR classifier.  A natural classifier to use in this setting is logistic regression, since it models the probability of a click-through event rather than returning a binary response, and when working with rare events, probabilistic predictions are useful.
# MAGIC 
# MAGIC First use [LogisticRegression](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression) from the pyspark.ml package to train a model using `oheTrainDF` with the given hyperparameter configuration.  `LogisticRegression` returns a [LogisticRegressionModel](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionModel).  Next, we'll use the `LogisticRegressionModel.coefficients` and `LogisticRegressionModel.intercept` attributes to print out some details of the model's parameters.  Note that these are the names of the object's attributes and should be called using a syntax like `model.coefficients` for a given `model`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
standardization = False
elasticNetParam = 0.0
regParam = .01
maxIter = 20

from pyspark.ml.classification import LogisticRegression
lr = (<FILL IN>)

lrModelBasic = <FILL IN>

print 'intercept: {0}'.format(lrModelBasic.intercept)
print 'length of coefficients: {0}'.format(len(lrModelBasic.coefficients))
sortedCoefficients = sorted(lrModelBasic.coefficients)[:5]

# COMMAND ----------

# TEST Logistic regression (4a)
Test.assertTrue(np.allclose(lrModelBasic.intercept,  -1.23621882418), 'incorrect value for model intercept')
Test.assertTrue(np.allclose(sortedCoefficients,
                [-0.11823093612565108, -0.10976569026754118, -0.10895403510742652, -0.10872819473509931,
                 -0.10461860693273944]), 'incorrect value for model coefficients')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4b) Log loss **
# MAGIC Throughout this lab, we will use log loss to evaluate the quality of models.  Log loss is defined as: \\[ \scriptsize \ell_{log}(p, y) = \begin{cases} -\log (p) & \text{if } y = 1 \\\ -\log(1-p) & \text{if } y = 0 \end{cases} \\] where \\( \scriptsize p\\) is a probability between 0 and 1 and \\( \scriptsize y\\) is a label of either 0 or 1. Log loss is a standard evaluation criterion when predicting rare-events such as click-through rate prediction (it is also the criterion used in the [Criteo Kaggle competition](https://www.kaggle.com/c/criteo-display-ad-challenge)).
# MAGIC 
# MAGIC Write a function `addLogLoss` to a DataFrame, and evaluate it on some sample inputs.  This does not require a UDF.  You can perform conditional branching with DataFrame columns using [when](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.when).

# COMMAND ----------

# Some example data
exampleLogLossDF = sqlContext.createDataFrame([(.5, 1), (.5, 0), (.99, 1), (.99, 0), (.01, 1),
                                               (.01, 0), (1., 1), (.0, 1), (1., 0)], ['p', 'label'])
exampleLogLossDF.show()

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import when, log, col
epsilon = 1e-16

def addLogLoss(df):
    """Computes and adds a 'logLoss' column to a DataFrame using 'p' and 'label' columns.

    Note:
        log(0) is undefined, so when p is 0 we add a small value (epsilon) to it and when
        p is 1 we subtract a small value (epsilon) from it.

    Args:
        df (DataFrame with 'p' and 'label' columns): A DataFrame with a probability column
            'p' and a 'label' column that corresponds to y in the log loss formula.

    Returns:
        DataFrame: A new DataFrame with an additional column called 'logLoss' where
    """
    return df.withColumn('logLoss', when(col('label') == 1, -log(col('p')+epsilon)).
                                    when(col('label') == 0, -log(1 - col('p') + epsilon)))

addLogLoss(exampleLogLossDF).show()

# COMMAND ----------

# TEST Log loss (4b)
logLossValues = addLogLoss(exampleLogLossDF).select('logLoss').rdd.map(lambda r: r[0]).collect()
Test.assertTrue(np.allclose(logLossValues[:-2],
                            [0.6931471805599451, 0.6931471805599451, 0.010050335853501338, 4.60517018598808,
                             4.605170185988081, 0.010050335853501338, -0.0]), 'computeLogLoss is not correct')
Test.assertTrue(not(any(map(lambda x: x is None, logLossValues[-2:]))),
                'computeLogLoss needs to bound p away from 0 and 1 by epsilon')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4c)  Baseline log loss **
# MAGIC 
# MAGIC Next we will use the function we wrote in Part (4b) to compute the baseline log loss on the training data. A very simple yet natural baseline model is one where we always make the same prediction independent of the given datapoint, setting the predicted value equal to the fraction of training points that correspond to click-through events (i.e., where the label is one). Compute this value (which is simply the mean of the training labels), and then use it to compute the training log loss for the baseline model.
# MAGIC 
# MAGIC > Note: you'll need to add a 'p' column to the `oheTrainDF` DataFrame so that it can be used in your function from Part (4b).  To represent a constant value as a column you can use the [lit](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.lit) function to wrap the value.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Note that our dataset has a very high click-through rate by design
# In practice click-through rate can be one to two orders of magnitude lower

from pyspark.sql.functions import lit
classOneFracTrain = (<FILL IN>)
print 'Training class one fraction = {0:.3f}'.format(classOneFracTrain)

logLossTrBase = (<FILL IN>)
print 'Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase)

# COMMAND ----------

# TEST Baseline log loss (4c)
Test.assertTrue(np.allclose(classOneFracTrain, 0.22522432452698143), 'incorrect value for classOneFracTrain')
Test.assertTrue(np.allclose(logLossTrBase, 0.5334411326715902), 'incorrect value for logLossTrBase')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4d) Predicted probability **
# MAGIC 
# MAGIC In order to compute the log loss for the model we trained in Part (4a), we need to write code to generate predictions from this model. Write a function that computes the raw linear prediction from this logistic regression model and then passes it through a [sigmoid function](http://en.wikipedia.org/wiki/Sigmoid_function) \\( \scriptsize \sigma(t) = (1+ e^{-t})^{-1} \\) to return the model's probabilistic prediction. Then compute probabilistic predictions on the training data.
# MAGIC 
# MAGIC Note that when incorporating an intercept into our predictions, we simply add the intercept to the value of the prediction obtained from the weights and features.  Alternatively, if the intercept was included as the first weight, we would need to add a corresponding feature to our data where the feature has the value one.  This is not the case here.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.types import DoubleType
from math import exp #  exp(-t) = e^-t

def addProbability(df, model):
    """Adds a probability column ('p') to a DataFrame given a model"""
    coefficientsBroadcast = sc.broadcast(model.coefficients)
    intercept = model.intercept

    def getP(features):
        """Calculate the probability for an observation given a set of weights and intercept.

        Note:
            We'll bound our raw prediction between 20 and -20 for numerical purposes.

        Args:
            x (SparseVector): A vector with values of 1.0 for features that exist in this
                observation and 0.0 otherwise.
            w (DenseVector): A vector of weights (betas) for the model.
            intercept (float): The model's intercept.

        Returns:
            float: A probability between 0 and 1.
        """
        # Compute the raw value
        rawPrediction = <FILL IN>
        # Bound the raw value between 20 and -20
        rawPrediction = <FILL IN>
        # Return the probability
        <FILL IN>

    getPUDF = udf(getP, DoubleType())
    return df.withColumn('p', getPUDF('features'))

addProbabilityModelBasic = lambda df: addProbability(df, lrModelBasic)
trainingPredictions = addProbabilityModelBasic(oheTrainDF).cache()

trainingPredictions.show(5)

# COMMAND ----------

# TEST Predicted probability (4d)
Test.assertTrue(np.allclose(trainingPredictions.selectExpr('sum(p)').first()[0], 18030.308138494933),
                'incorrect value for trainingPredictions')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4e) Evaluate the model **
# MAGIC 
# MAGIC We are now ready to evaluate the quality of the model we trained in Part (4a). To do this, first write a general function that takes as input a model and data, and outputs the log loss. Note that the log loss for multiple observations is the mean of the individual log loss values. Then run this function on the OHE training data, and compare the result with the baseline log loss.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def evaluateResults(df, model, baseline=None):
    """Calculates the log loss for the data given the model.

    Note:
        If baseline has a value the probability should be set to baseline before
        the log loss is calculated.  Otherwise, use addProbability to add the
        appropriate probabilities to the DataFrame.

    Args:
        df (DataFrame with 'label' and 'features' columns): A DataFrame containing
            labels and features.
        model (LogisticRegressionModel): A trained logistic regression model. This
            can be None if baseline is set.
        baseline (float): A baseline probability to use for the log loss calculation.

    Returns:
        float: Log loss for the data.
    """
    withProbabilityDF = <FILL IN>
    withLogLossDF = <FILL IN>
    logLoss = <FILL IN>
    return logLoss

logLossTrainModelBasic = evaluateResults(oheTrainDF, lrModelBasic)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTrBase, logLossTrainModelBasic))

# COMMAND ----------

# TEST Evaluate the model (4e)
Test.assertTrue(np.allclose(logLossTrainModelBasic, 0.4740881547541515),
                'incorrect value for logLossTrainModelBasic')
Test.assertTrue(np.allclose(evaluateResults(oheTrainDF, None,  0.5), 0.6931471805600546),
                'evaluateResults needs to handle baseline models')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4f) Validation log loss **
# MAGIC 
# MAGIC Next, using the `evaluateResults` function compute the validation log loss for both the baseline and logistic regression models. Notably, the baseline model for the validation data should still be based on the label fraction from the training dataset.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
logLossValBase = <FILL IN>

logLossValLR0 = <FILL IN>
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, logLossValLR0))

# COMMAND ----------

# TEST Validation log loss (4f)
Test.assertTrue(np.allclose(logLossValBase, 0.5395669919937838), 'incorrect value for logLossValBase')
Test.assertTrue(np.allclose(logLossValModelBasic, 0.48080532843294704),
                'incorrect value for logLossValModelBasic')

# COMMAND ----------

# MAGIC %md
# MAGIC **Visualization 2: ROC curve **
# MAGIC 
# MAGIC We will now visualize how well the model predicts our target.  To do this we generate a plot of the ROC curve.  The ROC curve shows us the trade-off between the false positive rate and true positive rate, as we liberalize the threshold required to predict a positive outcome.  A random model is represented by the dashed line.

# COMMAND ----------

labelsAndScores = addProbabilityModelBasic(oheValidationDF).select('label', 'p')
labelsAndWeights = labelsAndScores.collect()
labelsAndWeights.sort(key=lambda (k, v): v, reverse=True)
labelsByWeight = np.array([k for (k, v) in labelsAndWeights])

length = labelsByWeight.size
truePositives = labelsByWeight.cumsum()
numPositive = truePositives[-1]
falsePositives = np.arange(1.0, length + 1, 1.) - truePositives

truePositiveRate = truePositives / numPositive
falsePositiveRate = falsePositives / (length - numPositive)

# Generate layout and plot data
fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(falsePositiveRate, truePositiveRate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Part 5: Reduce feature dimension via feature hashing**

# COMMAND ----------

# MAGIC %md
# MAGIC ** (5a) Hash function **
# MAGIC 
# MAGIC As we just saw, using a one-hot-encoding featurization can yield a model with good statistical accuracy.  However, the number of distinct categories across all features is quite large -- recall that we observed 233K categories in the training data in Part (3c).  Moreover, the full Kaggle training dataset includes more than 33M distinct categories, and the Kaggle dataset itself is just a small subset of Criteo's labeled data.  Hence, featurizing via a one-hot-encoding representation would lead to a very large feature vector. To reduce the dimensionality of the feature space, we will use feature hashing.
# MAGIC 
# MAGIC Below is the hash function that we will use for this part of the lab.  We will first use this hash function with the three sample data points from Part (1a) to gain some intuition.  Specifically, run code to hash the three sample points using two different values for `numBuckets` and observe the resulting hashed feature dictionaries.

# COMMAND ----------

from collections import defaultdict
import hashlib

def hashFunction(rawFeats, numBuckets, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        numBuckets (int): Number of buckets to use as features.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = { category + ':' + str(ind):
                int(int(hashlib.md5(category + ':' + str(ind)).hexdigest(), 16) % numBuckets)
                for ind, category in rawFeats }
    if(printMapping): print mapping

    def mapUpdate(l, r):
        l[r] += 1.0
        return l

    sparseFeatures = reduce(mapUpdate, mapping.values(), defaultdict(float))
    return dict(sparseFeatures)

# Reminder of the sample values:
# sampleOne = [(0, 'mouse'), (1, 'black')]
# sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
# sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Use four buckets
sampOneFourBuckets = hashFunction(sampleOne, <FILL IN>, True)
sampTwoFourBuckets = hashFunction(sampleTwo, <FILL IN>, True)
sampThreeFourBuckets = hashFunction(sampleThree, <FILL IN>, True)

# Use one hundred buckets
sampOneHundredBuckets = hashFunction(sampleOne, <FILL IN>, True)
sampTwoHundredBuckets = hashFunction(sampleTwo, <FILL IN>, True)
sampThreeHundredBuckets = hashFunction(sampleThree, <FILL IN>, True)

print '\n\t\t 4 Buckets \t\t\t 100 Buckets'
print 'SampleOne:\t {0}\t\t\t {1}'.format(sampOneFourBuckets, sampOneHundredBuckets)
print 'SampleTwo:\t {0}\t\t {1}'.format(sampTwoFourBuckets, sampTwoHundredBuckets)
print 'SampleThree:\t {0}\t {1}'.format(sampThreeFourBuckets, sampThreeHundredBuckets)

# COMMAND ----------

# TEST Hash function (5a)
Test.assertEquals(sampOneFourBuckets, {3: 2.0}, 'incorrect value for sampOneFourBuckets')
Test.assertEquals(sampThreeHundredBuckets, {80: 1.0, 82: 1.0, 51: 1.0},
                  'incorrect value for sampThreeHundredBuckets')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (5b) Creating hashed features **
# MAGIC 
# MAGIC Next we will use this hash function to create hashed features for our CTR datasets. Use the provided UDF to create a function that takes in a DataFrame and returns labels and hashed features.  Then use this function to create new training, validation and test datasets with hashed features.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.mllib.linalg import Vectors
numHashBuckets = 2 ** 15

# UDF that returns a vector of hashed features given an Array of tuples
tuplesToHashFeaturesUDF = udf(lambda x: Vectors.sparse(numHashBuckets, hashFunction(x, numHashBuckets)), VectorUDT())

def addHashedFeatures(df):
    """Return a DataFrame with labels and hashed features.

    Note:
        Make sure to cache the DataFrame that you are returning.

    Args:
        df (DataFrame with 'tuples' column): A DataFrame containing the tuples to be hashed.

    Returns:
        DataFrame: A DataFrame with a 'label' column and a 'features' column that contains a
            SparseVector of hashed features.
    """
    <FILL IN>

hashTrainDF = <FILL IN>
hashValidationDF = <FILL IN>
hashTestDF = <FILL IN>

hashTrainDF.show()

# COMMAND ----------

# TEST Creating hashed features (5b)
hashTrainDFFeatureSum = sum(hashTrainDF
                              .rdd
                              .map(lambda r: sum(r[1].indices))
                              .take(10))
hashValidationDFFeatureSum = sum(hashValidationDF
                              .rdd
                              .map(lambda r: sum(r[1].indices))
                              .take(10))
hashTestDFFeatureSum = sum(hashTestDF
                              .rdd
                              .map(lambda r: sum(r[1].indices))
                              .take(10))

Test.assertEquals(hashTrainDFFeatureSum, 6643074, 'incorrect number of features in hashTrainDF')
Test.assertEquals(hashValidationDFFeatureSum, 6864079,
                  'incorrect number of features in hashValidationDF')
Test.assertEquals(hashTestDFFeatureSum, 6634666, 'incorrect number of features in hashTestDF')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (5c) Sparsity **
# MAGIC 
# MAGIC Since we have 33K hashed features versus 233K OHE features, we should expect OHE features to be sparser. Verify this hypothesis by computing the average sparsity of the OHE and the hashed training datasets.
# MAGIC 
# MAGIC Note that if you have a `SparseVector` named `sparse`, calling `len(sparse)` returns the total number of features, not the number features with entries.  `SparseVector` objects have the attributes `indices` and `values` that contain information about which features are nonzero.  Continuing with our example, these can be accessed using `sparse.indices` and `sparse.values`, respectively.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def vectorFeatureSparsity(sparseVector):
    """Calculates the sparsity of a SparseVector.

    Args:
        sparseVector (SparseVector): The vector containing the features.

    Returns:
        float: The ratio of features found in the vector to the total number of features.
    """
    <FILL IN>

featureSparsityUDF = udf(vectorFeatureSparsity, DoubleType())

aSparseVector = Vectors.sparse(5, {0: 1.0, 3: 1.0})
aSparseVectorSparsity = vectorFeatureSparsity(aSparseVector)
print 'This vector should have sparsity 2/5 or .4.'
print 'Sparsity = {0:.2f}.'.format(aSparseVectorSparsity)

# COMMAND ----------

# TEST Sparsity (5c)
Test.assertEquals(aSparseVectorSparsity, .4,
                'incorrect value for aSparseVectorSparsity')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (5d) Sparsity continued**
# MAGIC 
# MAGIC Now that we have a function to calculate vector sparsity, we'll wrap it in a UDF and apply it to an entire DataFrame to obtain the average sparsity for features in that DataFrame.  We'll use the function to find the average sparsity of the one-hot-encoded training DataFrame and of the hashed training DataFrame.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
featureSparsityUDF = udf(vectorFeatureSparsity, DoubleType())

def getSparsity(df):
    """Calculates the average sparsity for the features in a DataFrame.

    Args:
        df (DataFrame with 'features' column): A DataFrame with sparse features.

    Returns:
        float: The average feature sparsity.
    """
    return (<FILL IN>)

averageSparsityOHE = <FILL IN>
averageSparsityHash = <FILL IN>

print 'Average OHE Sparsity: {0:.7e}'.format(averageSparsityOHE)
print 'Average Hash Sparsity: {0:.7e}'.format(averageSparsityHash)

# COMMAND ----------

# TEST Sparsity (5d)
Test.assertTrue(np.allclose(averageSparsityOHE, 1.6670941e-04),
                'incorrect value for averageSparsityOHE')
Test.assertTrue(np.allclose(averageSparsityHash, 1.1896565e-03),
                'incorrect value for averageSparsityHash')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (5e) Logistic model with hashed features **
# MAGIC 
# MAGIC Now let's train a logistic regression model using the hashed training features. Use the hyperparameters provided, fit the model, and then evaluate the log loss on the training set.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
standardization = False
elasticNetParam = 0.7
regParam = .001
maxIter = 20

lrHash = (<FILL IN>)

lrModelHashed = <FILL IN>
print 'intercept: {0}'.format(lrModelHashed.intercept)
print len(lrModelHashed.coefficients)

logLossTrainModelHashed = <FILL IN>
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\thashed = {1:.3f}'
       .format(logLossTrBase, logLossTrainModelHashed))

# COMMAND ----------

# TEST Logistic model with hashed features (5e)
Test.assertTrue(np.allclose(logLossTrainModelHashed, 0.46517611883310084),
                'incorrect value for logLossTrainModelHashed')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (5f) Evaluate on the test set **
# MAGIC 
# MAGIC Finally, evaluate the model from Part (5e) on the test set.  Compare the resulting log loss with the baseline log loss on the test set, which can be computed in the same way that the validation log loss was computed in Part (4f).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Log loss for the best model from (5e)
logLossTest = <FILL IN>

# Log loss for the baseline model
classOneFracTest = <FILL IN>
print 'Class one fraction for test data: {0}'.format(classOneFracTest)
logLossTestBaseline = <FILL IN>

print ('Hashed Features Test Log Loss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTestBaseline, logLossTest))

# COMMAND ----------

# TEST Evaluate on the test set (5f)
Test.assertTrue(np.allclose(logLossTestBaseline, 0.5444498008367824),
                'incorrect value for logLossTestBaseline')
Test.assertTrue(np.allclose(logLossTest, 0.4713399075193271), 'incorrect value for logLossTest')

# COMMAND ----------

