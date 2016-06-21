# Databricks notebook source exported at Tue, 21 Jun 2016 01:35:40 UTC
# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC # **Word Count Lab: Building a word count application**
# MAGIC 
# MAGIC This lab will build on the techniques covered in the Spark tutorial to develop a simple word count application.  The volume of unstructured text in existence is growing dramatically, and Spark is an excellent tool for analyzing this type of data.  In this lab, we will write code that calculates the most common words in the [Complete Works of William Shakespeare](http://www.gutenberg.org/ebooks/100) retrieved from [Project Gutenberg](http://www.gutenberg.org/wiki/Main_Page).  This could also be scaled to larger applications, such as finding the most common words in Wikipedia.
# MAGIC 
# MAGIC ** During this lab we will cover: **
# MAGIC * *Part 1:* Creating a base DataFrame and performing operations
# MAGIC * *Part 2:* Counting with Spark SQL and DataFrames
# MAGIC * *Part 3:* Finding unique words and a mean value
# MAGIC * *Part 4:* Apply word count to a file
# MAGIC 
# MAGIC Note that for reference, you can look up the details of the relevant methods in [Spark's Python API](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.sql).

# COMMAND ----------

labVersion = 'cs105x-word-count-df-0.1.0'

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 1: Creating a base DataFrame and performing operations **

# COMMAND ----------

# MAGIC %md
# MAGIC In this part of the lab, we will explore creating a base DataFrame with `sqlContext.createDataFrame` and using DataFrame operations to count words.

# COMMAND ----------

# MAGIC %md
# MAGIC ** (1a) Create a DataFrame **
# MAGIC 
# MAGIC We'll start by generating a base DataFrame by using a Python list of tuples and the `sqlContext.createDataFrame` method.  Then we'll print out the type and schema of the DataFrame.  The Python API has several examples for using the [`createDataFrame` method](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SQLContext.createDataFrame).

# COMMAND ----------

wordsDF = sqlContext.createDataFrame([('cat',), ('elephant',), ('rat',), ('rat',), ('cat', )], ['word'])
wordsDF.show()
print type(wordsDF)
wordsDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ** (1b) Using DataFrame functions to add an 's' **
# MAGIC 
# MAGIC Let's create a new DataFrame from `wordsDF` by performing an operation that adds an 's' to each word.  To do this, we'll call the [`select` DataFrame function](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.select) and pass in a column that has the recipe for adding an 's' to our existing column.  To generate this `Column` object you should use the [`concat` function](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.concat) found in the [`pyspark.sql.functions` module](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions).  Note that `concat` takes in two or more string columns and returns a single string column.  In order to pass in a constant or literal value like 's', you'll need to wrap that value with the [`lit` column function](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.lit). 
# MAGIC 
# MAGIC Please replace `<FILL IN>` with your solution.  After you have created `pluralDF` you can run the next cell which contains two tests.  If you implementation is correct it will print `1 test passed` for each test.
# MAGIC  
# MAGIC This is the general form that exercises will take.  Exercises will include an explanation of what is expected, followed by code cells where one cell will have one or more `<FILL IN>` sections.  The cell that needs to be modified will have `# TODO: Replace <FILL IN> with appropriate code` on its first line.  Once the `<FILL IN>` sections are updated and the code is run, the test cell can then be run to verify the correctness of your solution.  The last code cell before the next markdown section will contain the tests.
# MAGIC 
# MAGIC > Note:
# MAGIC > Make sure that the resulting DataFrame has one column which is named 'word'.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import lit, concat

pluralDF = wordsDF.<FILL IN>
pluralDF.show()

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import lit, concat

pluralDF = wordsDF.select(concat('word', lit('s')).alias('word'))
pluralDF.show()

# COMMAND ----------

# Load in the testing code and check to see if your answer is correct
# If incorrect it will report back '1 test failed' for each failed test
# Make sure to rerun any cell you change before trying the test again
from databricks_test_helper import Test
# TEST Using DataFrame functions to add an 's' (1b)
Test.assertEquals(pluralDF.first()[0], 'cats', 'incorrect result: you need to add an s')
Test.assertEquals(pluralDF.columns, ['word'], "there should be one column named 'word'")

# COMMAND ----------

# PRIVATE_TEST Using DataFrame functions to add an 's' (1b)
Test.assertEquals(pluralDF.first()[0], 'cats', 'incorrect result: you need to add an s')
Test.assertEquals(pluralDF.columns, ['word'], "there should be one column named 'word'")

# COMMAND ----------

# MAGIC %md
# MAGIC ** (1c) Length of each word **
# MAGIC 
# MAGIC Now use the SQL `length` function to find the number of characters in each word.  The [`length` function](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.length) is found in the `pyspark.sql.functions` module.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import length
pluralLengthsDF = pluralDF.<FILL IN>
pluralLengthsDF.show()

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import length
pluralLengthsDF = pluralDF.select(length('word'))
pluralLengthsDF.show()

# COMMAND ----------

# TEST Length of each word (1e)
from collections import Iterable
asSelf = lambda v: map(lambda r: r[0] if isinstance(r, Iterable) and len(r) == 1 else r, v)

Test.assertEquals(asSelf(pluralLengthsDF.collect()), [4, 9, 4, 4, 4],
                  'incorrect values for pluralLengths')

# COMMAND ----------

# PRIVATE_TEST Length of each word (1e)
Test.assertEquals(asSelf(pluralLengthsDF.collect()), [4, 9, 4, 4, 4],
                  'incorrect values for pluralLengths')

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 2: Counting with Spark SQL and DataFrames **

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's count the number of times a particular word appears in the 'word' column. There are multiple ways to perform the counting, but some are much less efficient than others.
# MAGIC 
# MAGIC A naive approach would be to call `collect` on all of the elements and count them in the driver program. While this approach could work for small datasets, we want an approach that will work for any size dataset including terabyte- or petabyte-sized datasets. In addition, performing all of the work in the driver program is slower than performing it in parallel in the workers. For these reasons, we will use data parallel operations.

# COMMAND ----------

# MAGIC %md
# MAGIC ** (2a) Using `groupBy` and `count` **
# MAGIC 
# MAGIC Using DataFrames, we can preform aggregations by grouping the data using the [`groupBy` function](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.groupBy) on the DataFrame.  Using `groupBy` returns a [`GroupedData` object](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData) and we can use the functions available for `GroupedData` to aggregate the groups.  For example, we can call `avg` or `count` on a `GroupedData` object to obtain the average of the values in the groups or the number of occurrences in the groups, respectively.
# MAGIC 
# MAGIC To find the counts of words, group by the words and then use the [`count` function](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.count) to find the number of times that words occur.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
wordCountsDF = (wordsDF
                <FILL IN>)
wordCountsDF.show()

# COMMAND ----------

# ANSWER
wordCountsDF = (wordsDF
                .groupBy('word')
                .count())
wordCountsDF.show()

# COMMAND ----------

# TEST groupBy and count (2a)
Test.assertEquals(wordCountsDF.collect(), [('cat', 2), ('rat', 2), ('elephant', 1)],
                 'incorrect counts for wordCountsDF')

# COMMAND ----------

# PRIVATE_TEST groupBy and count (2a)
Test.assertEquals(wordCountsDF.collect(), [('cat', 2), ('rat', 2), ('elephant', 1)],
                 'incorrect counts for wordCountsDF')

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 3: Finding unique words and a mean value **

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3a) Unique words **
# MAGIC 
# MAGIC Calculate the number of unique words in `wordsDF`.  You can use other DataFrames that you have already created to make this easier.

# COMMAND ----------

from spark_notebook_helpers import printDataFrames

#This function returns all the DataFrames in the notebook and their corresponding column names.
printDataFrames(True)

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
uniqueWordsCount = <FILL IN>
print uniqueWordsCount

# COMMAND ----------

# ANSWER
uniqueWordsCount = wordCountsDF.count()
print uniqueWordsCount

# COMMAND ----------

# TEST Unique words (3a)
Test.assertEquals(uniqueWordsCount, 3, 'incorrect count of unique words')

# COMMAND ----------

# PRIVATE_TEST Unique words (3a)
Test.assertEquals(uniqueWordsCount, 3, 'incorrect count of unique words')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3b) Means of groups using DataFrames **
# MAGIC 
# MAGIC Find the mean number of occurrences of words in `wordCountsDF`.
# MAGIC 
# MAGIC You should use the [`mean` GroupedData method](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.mean) to accomplish this.  Note that when you use `groupBy` you don't need to pass in any columns.  A call without columns just prepares the DataFrame so that aggregation functions like `mean` can be applied.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
averageCount = (wordCountsDF
                <FILL IN>)

print averageCount

# COMMAND ----------

# ANSWER
averageCount = (wordCountsDF
                .groupBy()
                .mean()
                .first()[0])

print averageCount

# COMMAND ----------

# TEST Means of groups using DataFrames (3b)
Test.assertEquals(round(averageCount, 2), 1.67, 'incorrect value of averageCount')

# COMMAND ----------

# PRIVATE_TEST Means of groups using DataFrames (3b)
Test.assertEquals(round(averageCount, 2), 1.67, 'incorrect value of averageCount')

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 4: Apply word count to a file **

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we will finish developing our word count application.  We'll have to build the `wordCount` function, deal with real world problems like capitalization and punctuation, load in our data source, and compute the word count on the new data.

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4a) The `wordCount` function **
# MAGIC 
# MAGIC First, define a function for word counting.  You should reuse the techniques that have been covered in earlier parts of this lab.  This function should take in a DataFrame that is a list of words like `wordsDF` and return a DataFrame that has all of the words and their associated counts.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def wordCount(wordListDF):
    """Creates a DataFrame with word counts.

    Args:
        wordListDF (DataFrame of str): A DataFrame consisting of one string column called 'word'.

    Returns:
        DataFrame of (str, int): A DataFrame containing 'word' and 'count' columns.
    """
    return <FILL IN>

wordCount(wordsDF).show()

# COMMAND ----------

# ANSWER
def wordCount(wordListDF):
    """Creates a DataFrame with word counts.

    Args:
        wordListDF (DataFrame of str): A DataFrame consisting of one string column called 'word'.

    Returns:
        DataFrame of (str, int): A DataFrame containing 'word' and 'count' columns.
    """
    return (wordListDF
           .groupby('word')
           .count())

wordCount(wordsDF).show()

# COMMAND ----------

# TEST wordCount function (4a)
Test.assertEquals(sorted(wordCount(wordsDF).collect()),
                  [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect definition for wordCountDF function')

# COMMAND ----------

# PRIVATE_TEST wordCount function (4a)
privateWordsDF = (sc
                  .parallelize(['cat', 'cat', 'cat', 'rat', 'elephant', 'elephant'])
                  .map(lambda x: (x,))
                  .toDF(['word']))
Test.assertEquals(sorted(wordCount(privateWordsDF).collect()),
                  [('cat', 3), ('elephant', 2), ('rat', 1)],
                  'incorrect definition for wordCount function')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4b) Capitalization and punctuation **
# MAGIC 
# MAGIC Real world files are more complicated than the data we have been using in this lab. Some of the issues we have to address are:
# MAGIC   + Words should be counted independent of their capitialization (e.g., Spark and spark should be counted as the same word).
# MAGIC   + All punctuation should be removed.
# MAGIC   + Any leading or trailing spaces on a line should be removed.
# MAGIC  
# MAGIC Define the function `removePunctuation` that converts all text to lower case, removes any punctuation, and removes leading and trailing spaces.  Use the Python [regexp_replace](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.regexp_replace) module to remove any text that is not a letter, number, or space. If you are unfamiliar with regular expressions, you may want to review [this tutorial](https://developers.google.com/edu/python/regular-expressions) from Google.  Also, [this website](https://regex101.com/#python) is  a great resource for debugging your regular expression.
# MAGIC 
# MAGIC You should also use the `trim` and `lower` functions found in [pyspark.sql.functions](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions).
# MAGIC 
# MAGIC > Note that you shouldn't use any RDD operations or need to create custom user defined functions (udfs) to accomplish this task 

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import regexp_replace, trim, col, lower
def removePunctuation(column):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.

    Note:
        Only spaces, letters, and numbers should be retained.  Other characters should should be
        eliminated (e.g. it's becomes its).  Leading and trailing spaces should be removed after
        punctuation is removed.

    Args:
        column (Column): A Column containing a sentence.

    Returns:
        Column: A Column named 'sentence' with clean-up operations applied.
    """
    return <FILL IN>
  
sentenceDF = sqlContext.createDataFrame([('Hi, you!',), 
                                         (' No under_score!',), 
                                         (' *      Remove punctuation then spaces  * ',)], ['sentence'])
sentenceDF.show(truncate=False)
(sentenceDF
 .select(removePunctuation(col('sentence')))
 .show(truncate=False))

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import regexp_replace, trim, col, lower
def removePunctuation(column):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.

    Note:
        Only spaces, letters, and numbers should be retained.  Other characters should should be
        eliminated (e.g. it's becomes its).  Leading and trailing spaces should be removed after
        punctuation is removed.

    Args:
        column (Column): A Column containing a sentence.

    Returns:
        Column: A Column named 'sentence' with clean-up operations applied.
    """
    return trim(regexp_replace(lower(column), '[^a-z0-9 ]', '')).alias('sentence')
  
sentenceDF = sqlContext.createDataFrame([('Hi, you!',), 
                                         (' No under_score!',), 
                                         (' *      Remove punctuation then spaces  * ',)], ['sentence'])
sentenceDF.show(truncate=False)
(sentenceDF
 .select(removePunctuation(col('sentence')))
 .show(truncate=False))

# COMMAND ----------

# TEST Capitalization and punctuation (4b)
testPunctDF = sqlContext.createDataFrame([(" The Elephant's 4 cats. ",)])
Test.assertEquals(testPunctDF.select(removePunctuation(col('_1'))).first()[0],
                  'the elephants 4 cats',
                  'incorrect definition for removePunctuation function')

# COMMAND ----------

# PRIVATE_TEST Capitalization and punctuation (4b)
testPunctDF = sqlContext.createDataFrame([(" Hi, It's possible I'm cheating. ",)])
Test.assertEquals(testPunctDF.select(removePunctuation(col('_1'))).first()[0],
                  'hi its possible im cheating',
                  'incorrect definition for removePunctuation function')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4c) Load a text file **
# MAGIC 
# MAGIC For the next part of this lab, we will use the [Complete Works of William Shakespeare](http://www.gutenberg.org/ebooks/100) from [Project Gutenberg](http://www.gutenberg.org/wiki/Main_Page). To convert a text file into a DataFrame, we use the `sqlContext.read.text()` method. We also apply the recently defined `removePunctuation()` function using a `select()` transformation to strip out the punctuation and change all text to lower case.  Since the file is large we use `show(15)`, so that we only print 15 lines.

# COMMAND ----------

fileName = "dbfs:/databricks-datasets/cs100/lab1/data-001/shakespeare.txt"

shakespeareDF = sqlContext.read.text(fileName).select(removePunctuation(col('value')))
shakespeareDF.show(15, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4d) Words from lines **
# MAGIC 
# MAGIC Before we can use the `wordcount()` function, we have to address two issues with the format of the DataFrame:
# MAGIC   + The first issue is that  that we need to split each line by its spaces. 
# MAGIC   + The second issue is we need to filter out empty lines or words. 
# MAGIC  
# MAGIC Apply a transformation that will split each 'sentence' in the DataFrame by its spaces, and then transform from a DataFrame that contains lists of words into a DataFrame with each word in its own row.  To accomplish these two tasks you can use the `split` and `explode` functions found in [pyspark.sql.functions](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions). 
# MAGIC 
# MAGIC Once you have a DataFrame with one word per row you can apply the [DataFrame operation `where`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.where) to remove the rows that contain ''.
# MAGIC 
# MAGIC > Note that `shakeWordsDF` should be a DataFrame with one column named `word`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import split, explode
shakeWordsDF = (shakespeareDF
                <FILL IN>)

shakeWordsDF.show()
shakeWordsDFCount = shakeWordsDF.count()
print shakeWordsDFCount

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import split, size, explode
shakeWordsDF = (shakespeareDF
                .select(split('sentence', '\s+').alias('words'))
                .select(explode('words').alias('word'))
                .where(col('word') != ''))

shakeWordsDF.show()
shakeWordsDFCount = shakeWordsDF.count()
print shakeWordsDFCount

# COMMAND ----------

# TEST Remove empty elements (4d)
Test.assertEquals(shakeWordsDF.count(), 882996, 'incorrect value for shakeWordCount')
Test.assertEquals(shakeWordsDF.columns, ['word'], "shakeWordsDF should only contain the Column 'word'")

# COMMAND ----------

# PRIVATE_TEST Remove empty elements (4d)
Test.assertEquals(shakeWordsDF.count(), 882996, 'incorrect value for shakeWordCount')
Test.assertEquals(shakeWordsDF.columns, ['word'], "shakeWordsDF should only contain the Column 'word'")

# COMMAND ----------

# MAGIC %md
# MAGIC ** (4e) Count the words **
# MAGIC 
# MAGIC We now have a DataFrame that is only words.  Next, let's apply the `wordCount()` function to produce a list of word counts. We can view the first 20 words by using the `show()` action; however, we'd like to see the words in descending order of count, so we'll need to apply the [`orderBy` DataFrame method](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.orderBy) to first sort the DataFrame that is returned from `wordCount()`.
# MAGIC 
# MAGIC You'll notice that many of the words are common English words. These are called stopwords. In a later lab, we will see how to eliminate them from the results.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql.functions import desc
topWordsAndCountsDF = <FILL IN>
topWordsAndCountsDF.show()

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import desc
topWordsAndCountsDF = wordCount(shakeWordsDF).orderBy(desc('count'))
topWordsAndCountsDF.show()

# COMMAND ----------

# TEST Count the words (4e)
Test.assertEquals(topWordsAndCountsDF.take(15),
                  [(u'the', 27361), (u'and', 26028), (u'i', 20681), (u'to', 19150), (u'of', 17463),
                   (u'a', 14593), (u'you', 13615), (u'my', 12481), (u'in', 10956), (u'that', 10890),
                   (u'is', 9134), (u'not', 8497), (u'with', 7771), (u'me', 7769), (u'it', 7678)],
                  'incorrect value for top15WordsAndCountsDF')

# COMMAND ----------

# PRIVATE_TEST Count the words (4e)
Test.assertEquals(topWordsAndCountsDF.take(15),
                  [(u'the', 27361), (u'and', 26028), (u'i', 20681), (u'to', 19150), (u'of', 17463),
                   (u'a', 14593), (u'you', 13615), (u'my', 12481), (u'in', 10956), (u'that', 10890),
                   (u'is', 9134), (u'not', 8497), (u'with', 7771), (u'me', 7769), (u'it', 7678)],
                  'incorrect value for top15WordsAndCountsDF')

# COMMAND ----------

