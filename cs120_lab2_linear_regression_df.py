# Databricks notebook source exported at Tue, 19 Jul 2016 11:30:22 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC ![ML Logo](http://spark-mooc.github.io/web-assets/images/CS190.1x_Banner_300.png)
# MAGIC # Linear Regression Lab
# MAGIC 
# MAGIC This lab covers a common supervised learning pipeline, using a subset of the [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD). Our goal is to train a linear regression model to predict the release year of a song given a set of audio features.
# MAGIC 
# MAGIC ## This lab will cover:
# MAGIC *  *Part 1:* Read and parse the initial dataset
# MAGIC   * *Visualization 1:* Features
# MAGIC   * *Visualization 2:* Shifting labels
# MAGIC 
# MAGIC *  *Part 2:* Create and evaluate a baseline model
# MAGIC   * *Visualization 3:* Predicted vs. actual
# MAGIC 
# MAGIC *  *Part 3:* Train (via gradient descent) and evaluate a linear regression model
# MAGIC   * *Visualization 4:* Training error
# MAGIC 
# MAGIC *  *Part 4:* Train using SparkML and tune hyperparameters via grid search
# MAGIC   * *Visualization 5:* Best model's predictions
# MAGIC   * *Visualization 6:* Hyperparameter heat map
# MAGIC 
# MAGIC *  *Part 5:* Add interactions between features
# MAGIC 
# MAGIC > Note that, for reference, you can look up the details of:
# MAGIC > * the relevant Spark methods in [Spark's RDD Python API](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD) and [Spark's DataFrame Python API](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)
# MAGIC > * the relevant NumPy methods in the [NumPy Reference](http://docs.scipy.org/doc/numpy/reference/index.html)

# COMMAND ----------

labVersion = 'cs120x-lab2-1.0.5'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Read and parse the initial dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1a) Load and check the data
# MAGIC 
# MAGIC The raw data is currently stored in text file.  We will start by storing this raw data in as a DataFrame, with each element of the DataFrame representing a data point as a comma-delimited string. Each string starts with the label (a year) followed by numerical audio features. Use the DataFrame [count method](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.count) to check how many data points we have.  Then use the [take method](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.take) to create and print out a list of the first 5 data points in their initial string format.

# COMMAND ----------

# load testing library
from databricks_test_helper import Test
import os.path
file_name = os.path.join('databricks-datasets', 'cs190', 'data-001', 'millionsong.txt')

raw_data_df = sqlContext.read.load(file_name, 'text')

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
num_points = <FILL IN>
print num_points
sample_points = <FILL IN>
print sample_points

# COMMAND ----------

# TEST Load and check the data (1a)
Test.assertEquals(num_points, 6724, 'incorrect value for num_points')
Test.assertEquals(len(sample_points), 5, 'incorrect length for sample_points')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1b) Using `LabeledPoint`
# MAGIC 
# MAGIC In MLlib, labeled training instances are stored using the [LabeledPoint](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint) object.  Write the `parse_points` function that takes, as input, a DataFrame of comma-separated strings. We'll pass it the `raw_data_df` DataFrame.
# MAGIC 
# MAGIC It should parse each row in the DataFrame into individual elements, using Spark's [select](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.select) and [split](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.split) methods.
# MAGIC 
# MAGIC For example, split `"2001.0,0.884,0.610,0.600,0.474,0.247,0.357,0.344,0.33,0.600,0.425,0.60,0.419"` into `['2001.0', '0.884', '0.610', '0.600', '0.474', '0.247', '0.357', '0.344', '0.33', '0.600', '0.425', '0.60', '0.419']`.
# MAGIC 
# MAGIC The first value in the resulting list (`2001.0` in the example, above) is the label. The remaining values (`0.884`, `0.610`, etc., in the example) are the features.
# MAGIC 
# MAGIC After splitting each row, map it to a `LabeledPoint`. You'll have to step down to an RDD (using `.rdd`) or use a DataFrame user-defined function to convert to the `LabeledPoint` object. (See **Hint**, below.) If you step down to an RDD, you'll have to use [toDF()](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.toDF) to convert back to a DataFrame.
# MAGIC 
# MAGIC Use this new `parse_points` function to parse `raw_data_df`.  Then print out the features and label for the first training point, using the `features` and `label` attributes. Finally, calculate the number of features for this dataset.
# MAGIC 
# MAGIC ## Hint: Running Arbitrary Lambdas on a DataFrame
# MAGIC 
# MAGIC To solve this problem, you need a way to run your `parse_points` function on a DataFrame. There are two ways to do this, which we will illustrate with an extremely simple example.
# MAGIC 
# MAGIC Suppose you have a DataFrame consisting of a first name and a last name, and you want to add a unique [SHA-256](https://en.wikipedia.org/wiki/Secure_Hash_Algorithm) hash to each row.
# MAGIC 
# MAGIC ```
# MAGIC df = sqlContext.createDataFrame([("John", "Smith"), ("Ravi", "Singh"), ("Julia", "Jones")], ("first_name", "last_name"))
# MAGIC ```
# MAGIC 
# MAGIC Here's a simple function to calculate such a hash, using Python's built-in `hashlib` library:
# MAGIC 
# MAGIC ```
# MAGIC def make_hash(first_name, last_name):
# MAGIC     import hashlib
# MAGIC     m = hashlib.sha256()
# MAGIC     # Join the first name and last name by a blank and hash the resulting
# MAGIC     # string.
# MAGIC     full_name = ' '.join((first_name, last_name))
# MAGIC     m.update(full_name)
# MAGIC     return m.hexdigest()
# MAGIC ```
# MAGIC 
# MAGIC Okay, that's great. But, how do we use it on our DataFrame? We can use a UDF:
# MAGIC 
# MAGIC ```
# MAGIC from pyspark.sql.functions import udf
# MAGIC u_make_hash = udf(make_hash)
# MAGIC df2 = df.select(df['*'], u_make_hash(df['first_name'], df['last_name']))
# MAGIC # could run df2.show() here to prove it works
# MAGIC ```
# MAGIC 
# MAGIC Or we can step down to an RDD, use a lambda to call `make_hash` and have the lambda return a `Row` object, which Spark can use to ["infer" a new DataFrame](http://spark.apache.org/docs/latest/sql-programming-guide.html#inferring-the-schema-using-reflection).
# MAGIC 
# MAGIC ```
# MAGIC from pyspark.sql import Row
# MAGIC def make_hash_from_row(row):
# MAGIC     hash = make_hash(row[0], row[1])
# MAGIC     return Row(first_name=row[0], last_name=row[1], hash=hash)
# MAGIC 
# MAGIC df2 = (df.rdd
# MAGIC          .map(lambda row: make_hash_from_row(row))
# MAGIC          .toDF())
# MAGIC ```
# MAGIC 
# MAGIC These methods are roughly equivalent. You'll need to do something similar to convert _your_ `raw_data_df` DataFrame into a new DataFrame of `LabeledPoint` objects.

# COMMAND ----------

from pyspark.mllib.regression import LabeledPoint
import numpy as np

# Here is a sample raw data point:
# '2001.0,0.884,0.610,0.600,0.474,0.247,0.357,0.344,0.33,0.600,0.425,0.60,0.419'
# In this raw data point, 2001.0 is the label, and the remaining values are features

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql import functions as sql_functions

def parse_points(df):
    """Converts a DataFrame of comma separated unicode strings into a DataFrame of `LabeledPoints`.

    Args:
        df: DataFrame where each row is a comma separated unicode string. The first element in the string
            is the label and the remaining elements are the features.

    Returns:
        DataFrame: Each row is converted into a `LabeledPoint`, which consists of a label and
            features. To convert an RDD to a DataFrame, simply call toDF().
    """
    <FILL IN>

parsed_points_df = <FILL IN>
first_point_features = <FILL IN>
first_point_label = <FILL IN>
print first_point_features, first_point_label

d = len(first_point_features)
print d

# COMMAND ----------

# TEST Using LabeledPoint (1b)
Test.assertTrue(isinstance(first_point_label, float), 'label must be a float')
expectedX0 = [0.8841,0.6105,0.6005,0.4747,0.2472,0.3573,0.3441,0.3396,0.6009,0.4257,0.6049,0.4192]
Test.assertTrue(np.allclose(expectedX0, first_point_features, 1e-4, 1e-4),
                'incorrect features for firstPointFeatures')
Test.assertTrue(np.allclose(2001.0, first_point_label), 'incorrect label for firstPointLabel')
Test.assertTrue(d == 12, 'incorrect number of features')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 1: Features
# MAGIC 
# MAGIC First we will load and setup the visualization library. Then we will look at the raw features for 50 data points by generating a heatmap that visualizes each feature on a grey-scale and shows the variation of each feature across the 50 sample data points.  The features are all between 0 and 1, with values closer to 1 represented via darker shades of grey.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# takeSample(withReplacement, num, [seed]) randomly selects num elements from the dataset with/without replacement, and has an
# optional seed parameter that one can set for reproducible results

data_values = (parsed_points_df
               .rdd
               .map(lambda lp: lp.features.toArray())
               .takeSample(False, 50, 47))

# You can uncomment the line below to see randomly selected features.  These will be randomly
# selected each time you run the cell because there is no set seed.  Note that you should run
# this cell with the line commented out when answering the lab quiz questions.
# data_values = (parsedPointsDF
#                .rdd
#                .map(lambda lp: lp.features.toArray())
#                .takeSample(False, 50))

def prepare_plot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
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

# generate layout and plot
fig, ax = prepare_plot(np.arange(.5, 11, 1), np.arange(.5, 49, 1), figsize=(8,7), hideLabels=True,
                       gridColor='#eeeeee', gridWidth=1.1)
image = plt.imshow(data_values,interpolation='nearest', aspect='auto', cmap=cm.Greys)
for x, y, s in zip(np.arange(-.125, 12, 1), np.repeat(-.75, 12), [str(x) for x in range(12)]):
    plt.text(x, y, s, color='#999999', size='10')
plt.text(4.7, -3, 'Feature', color='#999999', size='11'), ax.set_ylabel('Observation')
display(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC ### (1c) Find the range
# MAGIC 
# MAGIC Now let's examine the labels to find the range of song years.  To do this, find the smallest and largest labels in the `parsed_points_df`.
# MAGIC 
# MAGIC We will use the min and max functions that are native to the DataFrames, and thus can be optimized using Spark's Catalyst Optimizer and Project Tungsten (don't worry about the technical details). This code will run faster than simply using the native min and max functions in Python. Use [selectExpr](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.selectExpr) to retrieve the min and max label values.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
content_stats = (parsed_points_df
                 .<FILL IN>)

min_year = <FILL IN>
max_year = <FILL IN>

print min_year, max_year

# COMMAND ----------

# TEST Find the range (1c)
Test.assertEquals(len(parsed_points_df.first().features), 12,
                  'unexpected number of features in sample point')
sum_feat_two = parsed_points_df.rdd.map(lambda lp: lp.features[2]).sum()
Test.assertTrue(np.allclose(sum_feat_two, 3158.96224351), 'parsedPointsDF has unexpected values')
year_range = max_year - min_year
Test.assertTrue(year_range == 89, 'incorrect range for minYear to maxYear')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1d) Shift labels
# MAGIC 
# MAGIC As we just saw, the labels are years in the 1900s and 2000s.  In learning problems, it is often natural to shift labels such that they start from zero.  Starting with `parsed_points_df`, create a new DataFrame in which the labels are shifted such that smallest label equals zero (hint: use `select`). After, use [withColumnRenamed](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.withColumnRenamed) to rename the appropriate columns to `features` and `label`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
parsed_data_df = parsed_points_df.<FILL IN>

# View the first point
print '\n{0}'.format(parsed_data_df.first())

# COMMAND ----------

# TEST Shift labels (1d)
old_sample_features = parsed_points_df.first().features
new_sample_features = parsed_data_df.first().features
Test.assertTrue(np.allclose(old_sample_features, new_sample_features),
                'new features do not match old features')
sum_feat_two = parsed_data_df.rdd.map(lambda lp: lp.features[2]).sum()
Test.assertTrue(np.allclose(sum_feat_two, 3158.96224351), 'parsed_data_df has unexpected values')
min_year_new = parsed_data_df.groupBy().min('label').first()[0]
max_year_new = parsed_data_df.groupBy().max('label').first()[0]
Test.assertTrue(min_year_new == 0, 'incorrect min year in shifted data')
Test.assertTrue(max_year_new == 89, 'incorrect max year in shifted data')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 2: Shifting labels
# MAGIC 
# MAGIC We will look at the labels before and after shifting them.  Both scatter plots below visualize tuples storing:
# MAGIC 
# MAGIC * a label value and
# MAGIC * the number of training points with this label.
# MAGIC 
# MAGIC The first scatter plot uses the initial labels, while the second one uses the shifted labels.  Note that the two plots look the same except for the labels on the x-axis.

# COMMAND ----------

# get data for plot
old_data = (parsed_points_df
             .rdd
             .map(lambda lp: (lp.label, 1))
             .reduceByKey(lambda x, y: x + y)
             .collect())
x, y = zip(*old_data)

# generate layout and plot data
fig, ax = prepare_plot(np.arange(1920, 2050, 20), np.arange(0, 150, 20))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel('Year'), ax.set_ylabel('Count')
display(fig)

# COMMAND ----------

# get data for plot
new_data = (parsed_points_df
             .rdd
             .map(lambda lp: (lp.label, 1))
             .reduceByKey(lambda x, y: x + y)
             .collect())
x, y = zip(*new_data)

# generate layout and plot data
fig, ax = prepare_plot(np.arange(0, 120, 20), np.arange(0, 120, 20))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel('Year (shifted)'), ax.set_ylabel('Count')
display(fig)
pass

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1e) Training, validation, and test sets
# MAGIC 
# MAGIC We're almost done parsing our dataset, and our final task involves spliting the dataset into training, validation and test sets. Use the [randomSplit method](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit) with the specified weights and seed to create DataFrames storing each of these datasets. Next, cache each of these DataFrames, as we will be accessing them multiple times in the remainder of this lab. Finally, compute the size of each dataset and verify that the sum of their sizes equals the value computed in Part (1a).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
weights = [.8, .1, .1]
seed = 42
parsed_train_data_df, parsed_val_data_df, parsed_test_data_df = parsed_data_df.<FILL IN>
parsed_train_data_df.<FILL IN>
parsed_val_data_df.<FILL IN>
parsed_test_data_df.<FILL IN>
n_train = parsed_train_data_df.<FILL IN>
n_val = parsed_val_data_df.<FILL IN>
n_test = parsed_test_data_df.<FILL IN>

print n_train, n_val, n_test, n_train + n_val + n_test
print parsed_data_df.count()

# COMMAND ----------

# TEST Training, validation, and test sets (1e)
Test.assertEquals(len(parsed_train_data_df.first().features), 12,
                  'parsed_train_data_df has wrong number of features')
sum_feat_two = (parsed_train_data_df
                 .rdd
                 .map(lambda lp: lp.features[2])
                 .sum())
sum_feat_three = (parsed_val_data_df
                  .rdd
                  .map(lambda lp: lp.features[3])
                  .reduce(lambda x, y: x + y))
sum_feat_four = (parsed_test_data_df
                  .rdd
                  .map(lambda lp: lp.features[4])
                  .reduce(lambda x, y: x + y))
Test.assertTrue(np.allclose([sum_feat_two, sum_feat_three, sum_feat_four],
                            2526.87757656, 297.340394298, 184.235876654),
                'parsed Train, Val, Test data has unexpected values')
Test.assertTrue(n_train + n_val + n_test == 6724, 'unexpected Train, Val, Test data set size')
Test.assertEquals(n_train, 5382, 'unexpected value for nTrain')
Test.assertEquals(n_val, 672, 'unexpected value for nVal')
Test.assertEquals(n_test, 670, 'unexpected value for nTest')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Create and evaluate a baseline model

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2a) Average label
# MAGIC 
# MAGIC A very simple yet natural baseline model is one where we always make the same prediction independent of the given data point, using the average label in the training set as the constant prediction value.  Compute this value, which is the average (shifted) song year for the training set.  Use `selectExpr` and `first()` from the [DataFrame API](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
average_train_year = (parsed_train_data_df
                        .<FILL IN>)
print average_train_year

# COMMAND ----------

# TEST Average label (2a)
Test.assertTrue(np.allclose(average_train_year, 54.0403195838),
                'incorrect value for average_train_year')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2b) Root mean squared error
# MAGIC 
# MAGIC We naturally would like to see how well this naive baseline performs.  We will use root mean squared error ([RMSE](http://en.wikipedia.org/wiki/Root-mean-square_deviation)) for evaluation purposes.  Using [Regression Evaluator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator),  compute the RMSE given a dataset of _(prediction, label)_ tuples.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.ml.evaluation import RegressionEvaluator

preds_and_labels = [(1., 3.), (2., 1.), (2., 2.)]
preds_and_labels_df = sqlContext.createDataFrame(preds_and_labels, ["prediction", "label"])

evaluator = RegressionEvaluator(<FILL IN>)
def calc_RMSE(dataset):
    """Calculates the root mean squared error for an dataset of (prediction, label) tuples.

    Args:
        dataset (DataFrame of (float, float)): A `DataFrame` consisting of (prediction, label) tuples.

    Returns:
        float: The square root of the mean of the squared errors.
    """
    return evaluator.<FILL IN>

example_rmse = calc_RMSE(preds_and_labels_df)
print example_rmse
# RMSE = sqrt[((1-3)^2 + (2-1)^2 + (2-2)^2) / 3] = 1.291

# COMMAND ----------

# TEST Root mean squared error (2b)
Test.assertTrue(np.allclose(example_rmse, 1.29099444874), 'incorrect value for exampleRMSE')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2c) Training, validation and test RMSE
# MAGIC 
# MAGIC Now let's calculate the training, validation and test RMSE of our baseline model. To do this, first create DataFrames of _(prediction, label)_ tuples for each dataset, and then call `calc_RMSE()`. Note that each RMSE can be interpreted as the average prediction error for the given dataset (in terms of number of years). You can use [createDataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SQLContext.createDataFrame) to make a DataFrame with the column names of "prediction" and "label" from an RDD.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
preds_and_labels_train = parsed_train_data_df.<FILL IN>
preds_and_labels_train_df = sqlContext.createDataFrame(preds_and_labels_train, ["prediction", "label"])
rmse_train_base = <FILL IN>

preds_and_labels_val = parsed_val_data_df.<FILL IN>
preds_and_labels_val_df = sqlContext.createDataFrame(preds_and_labels_val, ["prediction", "label"])
rmse_val_base = <FILL IN>

preds_and_labels_test = parsed_test_data_df.<FILL IN>
preds_and_labels_test_df = sqlContext.createDataFrame(preds_and_labels_test, ["prediction", "label"])
rmse_test_base = <FILL IN>

print 'Baseline Train RMSE = {0:.3f}'.format(rmse_train_base)
print 'Baseline Validation RMSE = {0:.3f}'.format(rmse_val_base)
print 'Baseline Test RMSE = {0:.3f}'.format(rmse_test_base)


# COMMAND ----------

# TEST Training, validation and test RMSE (2c)
Test.assertTrue(np.allclose([rmse_train_base, rmse_val_base, rmse_test_base],
                            [21.4303303309, 20.9179691056, 21.828603786]), 'incorrect RMSE values')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 3: Predicted vs. actual
# MAGIC 
# MAGIC We will visualize predictions on the validation dataset. The scatter plots below visualize tuples storing i) the predicted value and ii) true label.  The first scatter plot represents the ideal situation where the predicted value exactly equals the true label, while the second plot uses the baseline predictor (i.e., `average_train_year`) for all predicted values.  Further note that the points in the scatter plots are color-coded, ranging from light yellow when the true and predicted values are equal to bright red when they drastically differ.

# COMMAND ----------

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
cmap = get_cmap('YlOrRd')
norm = Normalize()

def squared_error(label, prediction):
    """Calculates the squared error for a single prediction."""
    return float((label - prediction)**2)

actual = np.asarray(parsed_val_data_df
                    .select('label')
                    .collect())
error = np.asarray(parsed_val_data_df
                   .rdd
                   .map(lambda lp: (lp.label, lp.label))
                   .map(lambda (l, p): squared_error(l, p))
                   .collect())
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 100, 20), np.arange(0, 100, 20))
plt.scatter(actual, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.5)
ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')
display(fig)

# COMMAND ----------

def squared_error(label, prediction):
    """Calculates the squared error for a single prediction."""
    return float((label - prediction)**2)

predictions = np.asarray(parsed_val_data_df
                         .rdd
                         .map(lambda lp: average_train_year)
                         .collect())
error = np.asarray(parsed_val_data_df
                   .rdd
                   .map(lambda lp: (lp.label, average_train_year))
                   .map(lambda (l, p): squared_error(l, p))
                   .collect())
norm = Normalize()
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = prepare_plot(np.arange(53.0, 55.0, 0.5), np.arange(0, 100, 20))
ax.set_xlim(53, 55)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.3)
ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Train (via gradient descent) and evaluate a linear regression model

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3a) Gradient summand
# MAGIC 
# MAGIC Now let's see if we can do better via linear regression, training a model via gradient descent (we'll omit the intercept for now). Recall that the gradient descent update for linear regression is:
# MAGIC \\[ \scriptsize \mathbf{w}_{i+1} = \mathbf{w}_i - \alpha_i \sum_j (\mathbf{w}_i^\top\mathbf{x}_j  - y_j) \mathbf{x}_j \,.\\]
# MAGIC where \\( \scriptsize i \\) is the iteration number of the gradient descent algorithm, and \\( \scriptsize j \\) identifies the observation.
# MAGIC 
# MAGIC First, implement a function that computes the summand for this update, i.e., the summand equals \\( \scriptsize (\mathbf{w}^\top \mathbf{x} - y) \mathbf{x} \, ,\\) and test out this function on two examples.  Use the `DenseVector` [dot](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.DenseVector.dot) method.

# COMMAND ----------

from pyspark.mllib.linalg import DenseVector

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def gradient_summand(weights, lp):
    """Calculates the gradient summand for a given weight and `LabeledPoint`.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        weights (DenseVector): An array of model weights (betas).
        lp (LabeledPoint): The `LabeledPoint` for a single observation.

    Returns:
        DenseVector: An array of values the same length as `weights`.  The gradient summand.
    """
    <FILL IN>

example_w = DenseVector([1, 1, 1])
example_lp = LabeledPoint(2.0, [3, 1, 4])
# gradient_summand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
summand_one = gradient_summand(example_w, example_lp)
print summand_one

example_w = DenseVector([.24, 1.2, -1.4])
example_lp = LabeledPoint(3.0, [-1.4, 4.2, 2.1])
summand_two = gradient_summand(example_w, example_lp)
print summand_two

# COMMAND ----------

# TEST Gradient summand (3a)
Test.assertTrue(np.allclose(summand_one, [18., 6., 24.]), 'incorrect value for summand_one')
Test.assertTrue(np.allclose(summand_two, [1.7304,-5.1912,-2.5956]), 'incorrect value for summand_two')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3b) Use weights to make predictions
# MAGIC 
# MAGIC Next, implement a `get_labeled_predictions` function that takes in weights and an observation's `LabeledPoint` and returns a _(prediction, label)_ tuple.  Note that we can predict by computing the dot product between weights and an observation's features.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def get_labeled_prediction(weights, observation):
    """Calculates predictions and returns a (prediction, label) tuple.

    Note:
        The labels should remain unchanged as we'll use this information to calculate prediction
        error later.

    Args:
        weights (np.ndarray): An array with one weight for each features in `trainData`.
        observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
            features for the data point.

    Returns:
        tuple: A (prediction, label) tuple. Convert the return type of the label and prediction to a float.
    """
    return <FILL IN>

weights = np.array([1.0, 1.5])
prediction_example = sc.parallelize([LabeledPoint(2, np.array([1.0, .5])),
                                     LabeledPoint(1.5, np.array([.5, .5]))])
preds_and_labels_example = prediction_example.map(lambda lp: get_labeled_prediction(weights, lp))
print preds_and_labels_example.collect()

# COMMAND ----------

# TEST Use weights to make predictions (3b)
Test.assertTrue(isinstance(preds_and_labels_example.first()[0], float), 'prediction must be a float')
Test.assertEquals(preds_and_labels_example.collect(), [(1.75, 2.0), (1.25, 1.5)],
                  'incorrect definition for getLabeledPredictions')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3c) Gradient descent
# MAGIC 
# MAGIC Next, implement a gradient descent function for linear regression and test out this function on an example.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
def linreg_gradient_descent(train_data, num_iters):
    """Calculates the weights and error for a linear regression model trained with gradient descent.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        train_data (RDD of LabeledPoint): The labeled data for use in training the model.
        num_iters (int): The number of iterations of gradient descent to perform.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (weights, training errors).  Weights will be the
            final weights (one weight per feature) for the model, and training errors will contain
            an error (RMSE) for each iteration of the algorithm.
    """
    # The length of the training data
    n = train_data.count()
    # The number of features in the training data
    d = len(train_data.first().features)
    w = np.zeros(d)
    alpha = 1.0
    # We will compute and store the training error after each iteration
    error_train = np.zeros(num_iters)
    for i in range(num_iters):
        # Use get_labeled_prediction from (3b) with trainData to obtain an RDD of (label, prediction)
        # tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
        # have large errors to start.
        preds_and_labels_train = <FILL IN>
        preds_and_labels_train_df = sqlContext.createDataFrame(preds_and_labels_train, ["prediction", "label"])
        error_train[i] = calc_RMSE(preds_and_labels_train_df)

        # Calculate the `gradient`.  Make use of the `gradient_summand` function you wrote in (3a).
        # Note that `gradient` should be a `DenseVector` of length `d`.
        gradient = <FILL IN>

        # Update the weights
        alpha_i = alpha / (n * np.sqrt(i+1))
        w -= <FILL IN>
    return w, error_train

# create a toy dataset with n = 10, d = 3, and then run 5 iterations of gradient descent
# note: the resulting model will not be useful; the goal here is to verify that
# linreg_gradient_descent is working properly
example_n = 10
example_d = 3
example_data = (sc
                 .parallelize(parsed_train_data_df.take(example_n))
                 .map(lambda lp: LabeledPoint(lp.label, lp.features[0:example_d])))
print example_data.take(2)
example_num_iters = 5
example_weights, example_error_train = linreg_gradient_descent(example_data, example_num_iters)
print example_weights

# COMMAND ----------

# TEST Gradient descent (3c)
expected_output = [22.68915382, 46.210194, 51.74336678]
Test.assertTrue(np.allclose(example_weights, expected_output), 'value of example_weights is incorrect')
expected_error = [66.32269596, 45.61098865, 38.6123992, 35.28952945, 33.4708604]
Test.assertTrue(np.allclose(example_error_train, expected_error),
                'value of exampleErrorTrain is incorrect')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3d) Train the model
# MAGIC 
# MAGIC Now let's train a linear regression model on all of our training data and evaluate its accuracy on the validation set.  Note that the test set will not be used here.  If we evaluated the model on the test set, we would bias our final results.
# MAGIC 
# MAGIC We've already done much of the required work: we computed the number of features in Part (1b); we created the training and validation datasets and computed their sizes in Part (1e); and, we wrote a function to compute RMSE in Part (2b).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
num_iters = 50
weights_LR0, error_train_LR0 = linreg_gradient_descent(<FILL IN>)

preds_and_labels = (parsed_val_data_df
                      .<FILL IN>)
preds_and_labels_df = sqlContext.createDataFrame(preds_and_labels, ["prediction", "label"])
rmse_val_LR0 = calc_RMSE(preds_and_labels_df)

print 'Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}'.format(rmse_val_base,
                                                                       rmse_val_LR0)

# COMMAND ----------

# TEST Train the model (3d)
expected_output = [ 22.2588534,   20.26005774,   0.01539014,   8.69071379,   5.63536339, -4.19700345,
                    15.54525224,   3.88968175,   9.76633157,   5.9276698,   11.41170336,   3.7525027 ]
Test.assertTrue(np.allclose(weights_LR0, expected_output), 'incorrect value for weights_LR0')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 4: Training error
# MAGIC 
# MAGIC We will look at the log of the training error as a function of iteration. The first scatter plot visualizes the logarithm of the training error for all 50 iterations.  The second plot shows the training error itself, focusing on the final 44 iterations.

# COMMAND ----------

norm = Normalize()
clrs = cmap(np.asarray(norm(np.log(error_train_LR0))))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 60, 10), np.arange(2, 6, 1))
ax.set_ylim(2, 6)
plt.scatter(range(0, num_iters), np.log(error_train_LR0), s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
ax.set_xlabel('Iteration'), ax.set_ylabel(r'$\log_e(errorTrainLR0)$')
display(fig)

# COMMAND ----------

norm = Normalize()
clrs = cmap(np.asarray(norm(error_train_LR0[6:])))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 60, 10), np.arange(17, 22, 1))
ax.set_ylim(17.8, 21.2)
plt.scatter(range(0, num_iters-6), error_train_LR0[6:], s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
ax.set_xticklabels(map(str, range(6, 66, 10)))
ax.set_xlabel('Iteration'), ax.set_ylabel(r'Training Error')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Train using SparkML and perform grid search

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4a) `LinearRegression`
# MAGIC 
# MAGIC We're already doing better than the baseline model, but let's see if we can do better by adding an intercept, using regularization, and (based on the previous visualization) training for more iterations.  SparkML's [LinearRegression](https://spark.apache.org/docs/1.6.1/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression) essentially implements the same algorithm that we implemented in Part (3b), albeit more efficiently and with various additional functionality, such as including an intercept in the model and allowing L1, L2, or [elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization). Elastic net regularization is a linear combination of L1 and L2 regularization. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
# MAGIC 
# MAGIC First use LinearRegression to train a model with [elastic net](https://spark.apache.org/docs/1.6.1/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression.elasticNetParam) regularization and an intercept.  This method returns a [LinearRegressionModel](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LinearRegressionModel).  Next, use the model's [coefficients](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegressionModel.coefficients) (weights) and [intercept](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegressionModel.intercept) attributes to print out the model's parameters.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
# Values to use when training the linear regression model

num_iters = 500  # iterations
reg = 1e-1  # regParam
alpha = .2  # elasticNetParam
use_intercept = True  # intercept

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
lin_reg = LinearRegression(<FILL IN>)
first_model = lin_reg.fit(parsed_train_data_df)

# coeffsLR1 stores the model coefficients; interceptLR1 stores the model intercept
coeffs_LR1 = <FILL IN>
intercept_LR1 = <FILL IN>
print coeffs_LR1, intercept_LR1

# COMMAND ----------

# TEST LinearRegression (4a)
expected_intercept = 64.2456893425
expected_weights = [21.8238800212, 27.6186877074, -66.4789086231, 54.191182811, -14.2978518435, -47.0287067393,35.1372526918,
                   -20.0165577186, 0.737339261177, -3.8022145328, -7.62277095338, -15.9836308238]
Test.assertTrue(np.allclose(intercept_LR1, expected_intercept), 'incorrect value for intercept_LR1')
Test.assertTrue(np.allclose(coeffs_LR1, expected_weights), 'incorrect value for weights_LR1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4b) Transform
# MAGIC 
# MAGIC Now use the [LinearRegressionModel.transform()](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegressionModel.transform) method to make predictions
# MAGIC on the `parsed_train_data_df`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
sample_prediction = first_model.<FILL IN>
display(sample_prediction)

# COMMAND ----------

# TEST Predict (4b)
Test.assertTrue(np.allclose(sample_prediction.first().prediction, 38.63757807530045),
                'incorrect value for sample_prediction')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4c) Evaluate RMSE
# MAGIC 
# MAGIC Next evaluate the accuracy of this model on the validation set.  Use the `transform()` method to create predictions, and then use the `calc_RMSE()` function from Part (2b).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
val_pred_df = <FILL IN>
rmse_val_LR1 = <FILL IN>

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}' +
       '\n\tLR1 = {2:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1)

# COMMAND ----------

# TEST Evaluate RMSE (4c)
Test.assertTrue(np.allclose(rmse_val_LR1, 15.3130800661), 'incorrect value for rmseValLR1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4d) Grid search
# MAGIC 
# MAGIC We're already outperforming the baseline on the validation set by almost 2 years on average, but let's see if we can do better. Perform grid search to find a good regularization parameter.  Try `regParam` values `1e-10`, `1e-5`, and `1.0`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
best_RMSE = rmse_val_LR1
best_reg_param = reg
best_model = first_model

num_iters = 500  # iterations
alpha = .2  # elasticNetParam
use_intercept = True  # intercept

for reg in <FILL IN>:
    lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
    model = lin_reg.fit(parsed_train_data_df)
    val_pred_df = model.transform(parsed_val_data_df)

    rmse_val_grid = calc_RMSE(val_pred_df)
    print rmse_val_grid

    if rmse_val_grid < best_RMSE:
        best_RMSE = rmse_val_grid
        best_reg_param = reg
        best_model = model

rmse_val_LR_grid = best_RMSE

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n' +
       '\tLRGrid = {3:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1, rmse_val_LR_grid)


# COMMAND ----------

# TEST Grid search (4d)
Test.assertTrue(np.allclose(15.3052663831, rmse_val_LR_grid), 'incorrect value for rmseValLRGrid')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 5: Best model's predictions
# MAGIC 
# MAGIC Next, we create a visualization similar to 'Visualization 3: Predicted vs. actual' from Part 2 using the predictions from the best model from Part (4d) on the validation dataset.  Specifically, we create a color-coded scatter plot visualizing tuples storing i) the predicted value from this model and ii) true label.

# COMMAND ----------

parsed_val_df = best_model.transform(parsed_val_data_df)
predictions = np.asarray(parsed_val_df
                         .select('prediction')
                         .collect())
actual = np.asarray(parsed_val_df
                      .select('label')
                      .collect())
error = np.asarray(parsed_val_df
                     .rdd
                     .map(lambda lp: squared_error(lp.label, lp.prediction))
                     .collect())

norm = Normalize()
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 120, 20), np.arange(0, 120, 20))
ax.set_xlim(15, 82), ax.set_ylim(-5, 105)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=.5)
ax.set_xlabel('Predicted'), ax.set_ylabel(r'Actual')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 6: Hyperparameter heat map
# MAGIC 
# MAGIC Next, we perform a visualization of hyperparameter search using a larger set of hyperparameters (with precomputed results).  Specifically, we create a heat map where the brighter colors correspond to lower RMSE values.  The first plot has a large area with brighter colors.  In order to differentiate within the bright region, we generate a second plot corresponding to the hyperparameters found within that region.

# COMMAND ----------

from matplotlib.colors import LinearSegmentedColormap

# Saved parameters and results, to save the time required to run 36 models
num_iters = 500
reg_params = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
alpha_params = [0.0, .1, .2, .4, .8, 1.0]
rmse_val = np.array([[ 15.317156766552452, 15.327211561989827, 15.357152971253697, 15.455092206273847, 15.73774335576239,
                       16.36423857334287, 15.315019185101972, 15.305949211619886, 15.355590337955194, 15.573049001631558,
                       16.231992712117222, 17.700179790697746, 15.305266383061921, 15.301104931027034, 15.400125020566225,
                       15.824676190630191, 17.045905140628836, 19.365558346037535, 15.292810983243772, 15.333756681057828,
                       15.620051033979871, 16.631757941340428, 18.948786862836954, 20.91796910560631, 15.308301384150049,
                       15.522394576046239, 16.414106221093316, 18.655978799189178, 20.91796910560631, 20.91796910560631,
                       15.33442896030322, 15.680134490745722, 16.86502909075323, 19.72915603626022, 20.91796910560631,
                       20.91796910560631 ]])

num_rows, num_cols = len(alpha_params), len(reg_params)
rmse_val = np.array(rmse_val)
rmse_val.shape = (num_rows, num_cols)

fig, ax = prepare_plot(np.arange(0, num_cols, 1), np.arange(0, num_rows, 1), figsize=(8, 7), hideLabels=True,
                       gridWidth=0.)
ax.set_xticklabels(reg_params), ax.set_yticklabels(alpha_params)
ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Alpha')

colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
image = plt.imshow(rmse_val,interpolation='nearest', aspect='auto',
                    cmap = colors)
display(fig)

# COMMAND ----------

# Zoom into the top left
alpha_params_zoom, reg_params_zoom = alpha_params[1:5], reg_params[:4]
rmse_val_zoom = rmse_val[1:5, :4]

num_rows, num_cols = len(alpha_params_zoom), len(reg_params_zoom)

fig, ax = prepare_plot(np.arange(0, num_cols, 1), np.arange(0, num_rows, 1), figsize=(8, 7), hideLabels=True,
                       gridWidth=0.)
ax.set_xticklabels(reg_params_zoom), ax.set_yticklabels(alpha_params_zoom)
ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Alpha')

colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
image = plt.imshow(rmse_val_zoom, interpolation='nearest', aspect='auto',
                    cmap = colors)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Add interactions between features

# COMMAND ----------

# MAGIC %md
# MAGIC ### (5a) Add 2-way interactions
# MAGIC 
# MAGIC So far, we've used the features as they were provided.  Now, we will add features that capture the two-way interactions between our existing features.  Write a function `two_way_interactions` that takes in a `LabeledPoint` and generates a new `LabeledPoint` that contains the old features and the two-way interactions between them.
# MAGIC 
# MAGIC > Note:
# MAGIC > * A dataset with three features would have nine ( \\( \scriptsize 3^2 \\) ) two-way interactions.
# MAGIC > * You might want to use [itertools.product](https://docs.python.org/2/library/itertools.html#itertools.product) to generate tuples for each of the possible 2-way interactions.
# MAGIC > * Remember that you can combine two `DenseVector` or `ndarray` objects using [np.hstack](http://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
import itertools

def two_way_interactions(lp):
    """Creates a new `LabeledPoint` that includes two-way interactions.

    Note:
        For features [x, y] the two-way interactions would be [x^2, x*y, y*x, y^2] and these
        would be appended to the original [x, y] feature list.

    Args:
        lp (LabeledPoint): The label and features for this observation.

    Returns:
        LabeledPoint: The new `LabeledPoint` should have the same label as `lp`.  Its features
            should include the features from `lp` followed by the two-way interaction features.
    """
    <FILL IN>

print two_way_interactions(LabeledPoint(0.0, [2, 3]))

# Transform the existing train, validation, and test sets to include two-way interactions.
# Remember to convert them back to DataFrames at the end.
train_data_interact_df = <FILL IN>
val_data_interact_df = <FILL IN>
test_data_interact_df = <FILL IN>

# COMMAND ----------

# TEST Add two-way interactions (5a)
two_way_example = two_way_interactions(LabeledPoint(0.0, [2, 3]))
Test.assertTrue(np.allclose(sorted(two_way_example.features),
                            sorted([2.0, 3.0, 4.0, 6.0, 6.0, 9.0])),
                'incorrect features generatedBy two_way_interactions')
two_way_point = two_way_interactions(LabeledPoint(1.0, [1, 2, 3]))
Test.assertTrue(np.allclose(sorted(two_way_point.features),
                            sorted([1.0,2.0,3.0,1.0,2.0,3.0,2.0,4.0,6.0,3.0,6.0,9.0])),
                'incorrect features generated by twoWayInteractions')
Test.assertEquals(two_way_point.label, 1.0, 'incorrect label generated by two_way_interactions')
Test.assertTrue(np.allclose(sum(train_data_interact_df.first().features), 28.623429648737346),
                'incorrect features in train_data_interact_df')
Test.assertTrue(np.allclose(sum(val_data_interact_df.first().features), 23.582959172640948),
                'incorrect features in val_data_interact_df')
Test.assertTrue(np.allclose(sum(test_data_interact_df.first().features), 26.045820467171758),
                'incorrect features in test_data_interact_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (5b) Build interaction model
# MAGIC 
# MAGIC Now, let's build the new model.  We've done this several times now.  To implement this for the new features, we need to change a few variable names.
# MAGIC 
# MAGIC  > Note:
# MAGIC  > * Remember that we should build our model from the training data and evaluate it on the validation data.
# MAGIC  > * You should re-run your hyperparameter search after changing features, as using the best hyperparameters from your prior model will not necessary lead to the best model.
# MAGIC  > * For this exercise, we have already preset the hyperparameters to reasonable values.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
num_iters = 500
reg = 1e-10
alpha = .2
use_intercept = True

lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
model_interact = lin_reg.fit(<FILL IN>)
preds_and_labels_interact_df = model_interact.transform(<FILL IN>)
rmse_val_interact = calc_RMSE(preds_and_labels_interact_df)

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n\tLRGrid = ' +
       '{3:.3f}\n\tLRInteract = {4:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1,
                                                 rmse_val_LR_grid, rmse_val_interact)

# COMMAND ----------

# TEST Build interaction model (5b)
Test.assertTrue(np.allclose(rmse_val_interact, 14.3495530997), 'incorrect value for rmse_val_interact')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (5c) Evaluate interaction model on test data
# MAGIC 
# MAGIC Our next step is to evaluate the new model on the test dataset.  Note that we haven't used the test set to evaluate any of our models.  Because of this, our evaluation provides us with an unbiased estimate for how our model will perform on new data.  If we had changed our model based on viewing its performance on the test set, our estimate of RMSE would likely be overly optimistic.
# MAGIC 
# MAGIC We'll also print the RMSE for both the baseline model and our new model.  With this information, we can see how much better our model performs than the baseline model.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
preds_and_labels_test_df = model_interact.<FILL IN>
rmse_test_interact = <FILL IN>

print ('Test RMSE:\n\tBaseline = {0:.3f}\n\tLRInteract = {1:.3f}'
       .format(rmse_test_base, rmse_test_interact))

# COMMAND ----------

# TEST Evaluate interaction model on test data (5c)
Test.assertTrue(np.allclose(rmse_test_interact, 14.9990015721),
                'incorrect value for rmse_test_interact')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (5d) Use a pipeline to create the interaction model
# MAGIC 
# MAGIC Our final step is to create the interaction model using a [Pipeline](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline).  Note that Spark contains the [PolynomialExpansion](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.PolynomialExpansion) transformer which will automatically generate interactions for us.  In this section, you'll need to generate the `PolynomialExpansion` transformer and set the stages for the `Pipeline` estimator.   Make sure to use a degree of 2 for `PolynomialExpansion`, set the input column appropriately, and set the output column to "polyFeatures".  The pipeline should contain two stages: the polynomial expansion and the linear regression.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.ml import Pipeline
from pyspark.ml.feature import PolynomialExpansion

num_iters = 500
reg = 1e-10
alpha = .2
use_intercept = True

polynomial_expansion = PolynomialExpansion(<FILL IN>)
linear_regression = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha,
                                     fitIntercept=use_intercept, featuresCol='polyFeatures')

pipeline = Pipeline(stages=[<FILL IN>])
pipeline_model = pipeline.fit(parsed_train_data_df)

predictions_df = pipeline_model.transform(parsed_test_data_df)

evaluator = RegressionEvaluator()
rmse_test_pipeline = evaluator.evaluate(predictions_df, {evaluator.metricName: "rmse"})
print('RMSE for test data set using pipelines: {0:.3f}'.format(rmse_test_pipeline))

# COMMAND ----------

# TEST Use a pipeline to create the interaction model (5d)
Test.assertTrue(np.allclose(rmse_test_pipeline, 14.99415450247963),
                'incorrect value for rmse_test_pipeline')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix A: Submitting Your Exercises to the Autograder
# MAGIC 
# MAGIC This section guides you through Step 2 of the grading process ("Submit to Autograder").
# MAGIC 
# MAGIC Once you confirm that your lab notebook is passing all tests, you can submit it first to the course autograder and then second to the edX website to receive a grade.
# MAGIC 
# MAGIC ** Note that you can only submit to the course autograder once every 1 minute. **

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(a): Restart your cluster by clicking on the dropdown next to your cluster name and selecting "Restart Cluster".
# MAGIC 
# MAGIC You can do this step in either notebook, since there is one cluster for your notebooks.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/submit_restart.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(b): _IN THIS NOTEBOOK_, click on "Run All" to run all of the cells.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/submit_runall.png" alt="Drawing" style="height: 80px"/>
# MAGIC 
# MAGIC This step will take some time.
# MAGIC 
# MAGIC Wait for your cluster to finish running the cells in your lab notebook before proceeding.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(c): Publish this notebook
# MAGIC 
# MAGIC Publish _this_ notebook by clicking on the "Publish" button at the top.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish0.png" alt="Drawing" style="height: 150px"/>
# MAGIC 
# MAGIC When you click on the button, you will see the following popup.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish1.png" alt="Drawing" />
# MAGIC 
# MAGIC When you click on "Publish", you will see a popup with your notebook's public link. **Copy the link and set the `notebook_URL` variable in the AUTOGRADER notebook (not this notebook).**
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish2.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(d): Set the notebook URL and Lab ID in the Autograder notebook, and run it
# MAGIC 
# MAGIC Go to the Autograder notebook and paste the link you just copied into it, so that it is assigned to the `notebook_url` variable.
# MAGIC 
# MAGIC ```
# MAGIC notebook_url = "..." # put your URL here
# MAGIC ```
# MAGIC 
# MAGIC Then, find the line that looks like this:
# MAGIC 
# MAGIC ```
# MAGIC lab = <FILL IN>
# MAGIC ```
# MAGIC and change `<FILL IN>` to "CS120x-lab2":
# MAGIC 
# MAGIC ```
# MAGIC lab = "CS120x-lab2"
# MAGIC ```
# MAGIC 
# MAGIC Then, run the Autograder notebook to submit your lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ### <img src="http://spark-mooc.github.io/web-assets/images/oops.png" style="height: 200px"/> If things go wrong
# MAGIC 
# MAGIC It's possible that your notebook looks fine to you, but fails in the autograder. (This can happen when you run cells out of order, as you're working on your notebook.) If that happens, just try again, starting at the top of Appendix A.

# COMMAND ----------


