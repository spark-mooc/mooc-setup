# Databricks notebook source exported at Fri, 19 Aug 2016 18:07:13 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/movie-camera.png" style="float:right; height: 200px; margin: 10px; border: 1px solid #ddd; border-radius: 15px 15px 15px 15px; padding: 10px"/>
# MAGIC 
# MAGIC # Predicting Movie Ratings
# MAGIC 
# MAGIC One of the most common uses of big data is to predict what users want.  This allows Google to show you relevant ads, Amazon to recommend relevant products, and Netflix to recommend movies that you might like.  This lab will demonstrate how we can use Apache Spark to recommend movies to a user.  We will start with some basic techniques, and then use the [Spark MLlib][mllib] library's Alternating Least Squares method to make more sophisticated predictions.
# MAGIC 
# MAGIC For this lab, we will use a subset dataset of 20 million ratings. This dataset is pre-mounted on Databricks and is from the [MovieLens stable benchmark rating dataset](http://grouplens.org/datasets/movielens/). However, the same code you write will also work on the full dataset (though running with the full dataset on Community Edition is likely to take quite a long time).
# MAGIC 
# MAGIC In this lab:
# MAGIC * *Part 0*: Preliminaries
# MAGIC * *Part 1*: Basic Recommendations
# MAGIC * *Part 2*: Collaborative Filtering
# MAGIC * *Part 3*: Predictions for Yourself
# MAGIC 
# MAGIC As mentioned during the first Learning Spark lab, think carefully before calling `collect()` on any datasets.  When you are using a small dataset, calling `collect()` and then using Python to get a sense for the data locally (in the driver program) will work fine, but this will not work when you are using a large dataset that doesn't fit in memory on one machine.  Solutions that call `collect()` and do local analysis that could have been done with Spark will likely fail in the autograder and not receive full credit.
# MAGIC [mllib]: https://spark.apache.org/mllib/

# COMMAND ----------

labVersion = 'cs110x.lab2-1.0.0'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code
# MAGIC 
# MAGIC This assignment can be completed using basic Python and pySpark DataFrame Transformations and Actions.  Libraries other than math are not necessary. With the exception of the ML functions that we introduce in this assignment, you should be able to complete all parts of this homework using only the Spark functions you have used in prior lab exercises (although you are welcome to use more features of Spark if you like!).
# MAGIC 
# MAGIC We'll be using motion picture data, the same data last year's CS100.1x used. However, in this course, we're using DataFrames, rather than RDDs.
# MAGIC 
# MAGIC The following cell defines the locations of the data files. If you want to run an exported version of this lab on your own machine (i.e., outside of Databricks), you'll need to download your own copy of the 20-million movie data set, and you'll need to adjust the paths, below.
# MAGIC 
# MAGIC **To Do**: Run the following cell.

# COMMAND ----------

import os
from databricks_test_helper import Test

dbfs_dir = '/databricks-datasets/cs110x/ml-20m/data-001'
ratings_filename = dbfs_dir + '/ratings.csv'
movies_filename = dbfs_dir + '/movies.csv'

# The following line is here to enable this notebook to be exported as source and
# run on a local machine with a local copy of the files. Just change the dbfs_dir,
# above.
if os.path.sep != '/':
  # Handle Windows.
  ratings_filename = ratings_filename.replace('/', os.path.sep)
  movie_filename = movie_filename.replace('/', os.path.sep)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 0: Preliminaries
# MAGIC 
# MAGIC We read in each of the files and create a DataFrame consisting of parsed lines.
# MAGIC 
# MAGIC ### The 20-million movie sample
# MAGIC 
# MAGIC The 20-million movie sample consists of CSV files (with headers), so there's no need to parse the files manually, as Spark CSV can do the job.

# COMMAND ----------

# MAGIC %md
# MAGIC First, let's take a look at the directory containing our files.

# COMMAND ----------

display(dbutils.fs.ls(dbfs_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CPU vs I/O tradeoff
# MAGIC 
# MAGIC Note that we have both compressed files (ending in `.gz`) and uncompressed files. We have a CPU vs. I/O tradeoff here. If I/O is the bottleneck, then we want to process the compressed files and pay the extra CPU overhead. If CPU is the bottleneck, then it makes more sense to process the uncompressed files.
# MAGIC 
# MAGIC We've done some experiments, and we've determined that CPU more of a bottleneck than I/O, on Community Edition. So, we're going to process the uncompressed data. In addition, we're going to speed things up further by specifying the DataFrame schema explicitly. (When the Spark CSV adapter infers the schema from a CSV file, it has to make an extra pass over the file. That'll slow things down here, and it isn't really necessary.)
# MAGIC 
# MAGIC **To Do**: Run the following cell, which will define the schemas.

# COMMAND ----------

from pyspark.sql.types import *

ratings_df_schema = StructType(
  [StructField('userId', IntegerType()),
   StructField('movieId', IntegerType()),
   StructField('rating', DoubleType())]
)
movies_df_schema = StructType(
  [StructField('ID', IntegerType()),
   StructField('title', StringType())]
)

# COMMAND ----------

# MAGIC %md
# MAGIC **To Do**: Run the following cell to load the data. Please be patient: The code about 30 seconds to run.

# COMMAND ----------

from pyspark.sql.functions import regexp_extract
from pyspark.sql.types import *

raw_ratings_df = sqlContext.read.format('com.databricks.spark.csv').options(header=True, inferSchema=False).schema(ratings_df_schema).load(ratings_filename)
ratings_df = raw_ratings_df.drop('Timestamp').cache()

raw_movies_df = sqlContext.read.format('com.databricks.spark.csv').options(header=True, inferSchema=False).schema(movies_df_schema).load(movies_filename)
movies_df = raw_movies_df.drop('Genres').withColumnRenamed('movieId', 'ID').cache()

raw_ratings_count = raw_ratings_df.count()
ratings_count = ratings_df.count()
raw_movies_count = raw_movies_df.count()
movies_count = movies_df.count()

print 'There are %s ratings and %s movies in the datasets' % (ratings_count, movies_count)
print 'Ratings:'
ratings_df.show(3)
print 'Movies:'
movies_df.show(3, truncate=False)

assert raw_ratings_count == ratings_count
assert raw_movies_count == movies_count


# COMMAND ----------

# MAGIC %md
# MAGIC ### Caching
# MAGIC 
# MAGIC The Databricks File System (DBFS) sits on top of S3. We're going to be accessing this data a lot. Rather than read it over and over again from S3, let's cache both
# MAGIC the movies DataFrame and the ratings DataFrame in memory.
# MAGIC 
# MAGIC **To Do**: Run the following cell.

# COMMAND ----------

ratings_df.cache()
movies_df.cache()

assert ratings_df.is_cached
assert movies_df.is_cached

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's do a quick verification of the data.
# MAGIC 
# MAGIC **To do**: Run the following cell. It should run without errors.

# COMMAND ----------

assert ratings_count == 20000263
assert movies_count == 27278
assert movies_df.filter(movies_df.title == 'Toy Story (1995)').count() == 1
assert ratings_df.filter((ratings_df.userId == 6) & (ratings_df.movieId == 1) & (ratings_df.rating == 5.0)).count() == 1

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a quick look at some of the data in the two DataFrames.
# MAGIC 
# MAGIC **To Do**: Run the following two cells.

# COMMAND ----------

display(movies_df)

# COMMAND ----------

display(ratings_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Basic Recommendations
# MAGIC 
# MAGIC One way to recommend movies is to always recommend the movies with the highest average rating. In this part, we will use Spark to find the name, number of ratings, and the average rating of the 20 movies with the highest average rating and more than 500 reviews. We want to filter our movies with high ratings but fewer than or equal to 500 reviews because movies with few reviews may not have broad appeal to everyone.

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1a) Movies with Highest Average Ratings
# MAGIC 
# MAGIC Now that we have a way to calculate the average ratings, we will use the `getCountsAndAverages()` helper function with Spark to determine movies with highest average ratings.
# MAGIC 
# MAGIC The steps you should perform are:
# MAGIC 
# MAGIC 1. Recall that the `ratings_df` contains three columns:
# MAGIC     - The ID of the user who rated the film
# MAGIC     - the ID of the movie being rated
# MAGIC     - and the rating.
# MAGIC 
# MAGIC    First, transform `ratings_df` into a second DataFrame, `movie_ids_with_avg_ratings`, with the following columns:
# MAGIC     - The movie ID
# MAGIC     - The number of ratings for the movie
# MAGIC     - The average of all the movie's ratings
# MAGIC 
# MAGIC 2. Transform `movie_ids_with_avg_ratings` to another DataFrame, `movie_names_with_avg_ratings_df` that adds the movie name to each row. `movie_names_with_avg_ratings_df`
# MAGIC    will contain these columns:
# MAGIC     - The movie ID
# MAGIC     - The movie name
# MAGIC     - The number of ratings for the movie
# MAGIC     - The average of all the movie's ratings
# MAGIC 
# MAGIC    **Hint**: You'll need to do a join.
# MAGIC 
# MAGIC You should end up with something like the following:
# MAGIC ```
# MAGIC movie_ids_with_avg_ratings_df:
# MAGIC +-------+-----+------------------+
# MAGIC |movieId|count|average           |
# MAGIC +-------+-----+------------------+
# MAGIC |1831   |7463 |2.5785207021305103|
# MAGIC |431    |8946 |3.695059244355019 |
# MAGIC |631    |2193 |2.7273141814865483|
# MAGIC +-------+-----+------------------+
# MAGIC only showing top 3 rows
# MAGIC 
# MAGIC movie_names_with_avg_ratings_df:
# MAGIC +-------+-----------------------------+-----+-------+
# MAGIC |average|title                        |count|movieId|
# MAGIC +-------+-----------------------------+-----+-------+
# MAGIC |5.0    |Ella Lola, a la Trilby (1898)|1    |94431  |
# MAGIC |5.0    |Serving Life (2011)          |1    |129034 |
# MAGIC |5.0    |Diplomatic Immunity (2009? ) |1    |107434 |
# MAGIC +-------+-----------------------------+-----+-------+
# MAGIC only showing top 3 rows
# MAGIC ```

# COMMAND ----------

# TODO: Replace <FILL_IN> with appropriate code
from pyspark.sql import functions as F

# From ratingsDF, create a movie_ids_with_avg_ratings_df that combines the two DataFrames
movie_ids_with_avg_ratings_df = ratings_df.groupBy('movieId').agg(F.count(ratings_df.rating).alias("count"), F.avg(ratings_df.rating).alias("average"))
print 'movie_ids_with_avg_ratings_df:'
movie_ids_with_avg_ratings_df.show(3, truncate=False)

movie_name_df = movie_ids_with_avg_ratings_df.<FILL_IN>
movie_name_with_avg_ratings_df = movie_name_df.<FILL_IN>

print 'movie_name_with_avg_ratings_df:'
movie_name_with_avg_ratings_df.show(3, truncate=False)

# COMMAND ----------

# TEST
Test.assertEquals(movie_ids_with_avg_ratings_df.count(), 26744,
                'incorrect movie_ids_with_avg_ratings_df.count() (expected 3706)')
movie_ids_with_ratings_take_ordered = movie_ids_with_avg_ratings_df.orderBy('MovieID').take(3)
_take_0 = movie_ids_with_ratings_take_ordered[0]
_take_1 = movie_ids_with_ratings_take_ordered[1]
_take_2 = movie_ids_with_ratings_take_ordered[2]
Test.assertTrue(_take_0[0] == 1 and _take_0[1] == 49695,
                'incorrect count of ratings for movie with ID {0} (expected 993)'.format(_take_0[0]))
Test.assertEquals(round(_take_0[2], 2), 3.92, "Incorrect average for movie ID {0}. Expected 3.92".format(_take_0[0]))

Test.assertTrue(_take_1[0] == 2 and _take_1[1] == 22243,
                'incorrect count of ratings for movie with ID {0} (expected 22243)'.format(_take_1[0]))
Test.assertEquals(round(_take_1[2], 2), 3.21, "Incorrect average for movie ID {0}. Expected 3.21".format(_take_1[0]))

Test.assertTrue(_take_2[0] == 3 and _take_2[1] == 12735,
                'incorrect count of ratings for movie with ID {0} (expected 12735)'.format(_take_2[0]))
Test.assertEquals(round(_take_2[2], 2), 3.15, "Incorrect average for movie ID {0}. Expected 3.15".format(_take_2[0]))


Test.assertEquals(movie_names_with_avg_ratings_df.count(), 26744,
                  'incorrect movie_names_with_avg_ratings_df.count() (expected 3615)')
movie_names_with_ratings_take_ordered = movie_names_with_avg_ratings_df.orderBy(['average', 'title']).take(3)
result = [(r['average'], r['title'], r['count'], r['movieId']) for r in movie_names_with_ratings_take_ordered]
Test.assertEquals(result,
                  [(0.5, u'13 Fighting Men (1960)', 1, 109355),
                   (0.5, u'20 Years After (2008)', 1, 131062),
                   (0.5, u'3 Holiday Tails (Golden Christmas 2: The Second Tail, A) (2011)', 1, 111040)],
                  'incorrect top 3 entries in movie_names_with_avg_ratings_df')


# COMMAND ----------

# MAGIC %md
# MAGIC ### (1b) Movies with Highest Average Ratings and more than 500 reviews
# MAGIC 
# MAGIC Now that we have a DataFrame of the movies with highest average ratings, we can use Spark to determine the 20 movies with highest average ratings and more than 500 reviews.
# MAGIC 
# MAGIC Add a single DataFrame transformation (in place of `<FILL_IN>`, below) to limit the results to movies with ratings from more than 500 people.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
movies_with_500_ratings_or_more = movie_names_with_avg_ratings_df.<FILL_IN>
print 'Movies with highest ratings:'
movies_with_500_ratings_or_more.show(20, truncate=False)

# COMMAND ----------

# TEST Movies with Highest Average Ratings and more than 500 Reviews (1c)

Test.assertEquals(movies_with_500_ratings_or_more.count(), 4483,
                  'incorrect movies_with_500_ratings_or_more.count(). Expected 4483.')
top_20_results = [(r['average'], r['title'], r['count']) for r in movies_with_500_ratings_or_more.orderBy(F.desc('average')).take(20)]

Test.assertEquals(top_20_results,
                  [(4.446990499637029, u'Shawshank Redemption, The (1994)', 63366),
                   (4.364732196832306, u'Godfather, The (1972)', 41355),
                   (4.334372207803259, u'Usual Suspects, The (1995)', 47006),
                   (4.310175010988133, u"Schindler's List (1993)", 50054),
                   (4.275640557704942, u'Godfather: Part II, The (1974)', 27398),
                   (4.2741796572216, u'Seven Samurai (Shichinin no samurai) (1954)', 11611),
                   (4.271333600779414, u'Rear Window (1954)', 17449),
                   (4.263182346109176, u'Band of Brothers (2001)', 4305),
                   (4.258326830670664, u'Casablanca (1942)', 24349),
                   (4.256934865900383, u'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)', 6525),
                   (4.24807897901911, u"One Flew Over the Cuckoo's Nest (1975)", 29932),
                   (4.247286821705426, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)', 23220),
                   (4.246001523229246, u'Third Man, The (1949)', 6565),
                   (4.235410064157069, u'City of God (Cidade de Deus) (2002)', 12937),
                   (4.2347902097902095, u'Lives of Others, The (Das leben der Anderen) (2006)', 5720),
                   (4.233538107122288, u'North by Northwest (1959)', 15627),
                   (4.2326233183856505, u'Paths of Glory (1957)', 3568),
                   (4.227123123722136, u'Fight Club (1999)', 40106),
                   (4.224281931146873, u'Double Indemnity (1944)', 4909),
                   (4.224137931034483, u'12 Angry Men (1957)', 12934)],
                  'Incorrect top 20 movies with 500 or more ratings')


# COMMAND ----------

# MAGIC %md
# MAGIC Using a threshold on the number of reviews is one way to improve the recommendations, but there are many other good ways to improve quality. For example, you could weight ratings by the number of ratings.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Collaborative Filtering
# MAGIC In this course, you have learned about many of the basic transformations and actions that Spark allows us to apply to distributed datasets.  Spark also exposes some higher level functionality; in particular, Machine Learning using a component of Spark called [MLlib][mllib].  In this part, you will learn how to use MLlib to make personalized movie recommendations using the movie data we have been analyzing.
# MAGIC 
# MAGIC <img src="https://courses.edx.org/c4x/BerkeleyX/CS100.1x/asset/Collaborative_filtering.gif" alt="collaborative filtering" style="float: right"/>
# MAGIC 
# MAGIC We are going to use a technique called [collaborative filtering][collab]. Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a person chosen randomly. You can read more about collaborative filtering [here][collab2].
# MAGIC 
# MAGIC The image at the right (from [Wikipedia][collab]) shows an example of predicting of the user's rating using collaborative filtering. At first, people rate different items (like videos, images, games). After that, the system is making predictions about a user's rating for an item, which the user has not rated yet. These predictions are built upon the existing ratings of other users, who have similar ratings with the active user. For instance, in the image below the system has made a prediction, that the active user will not like the video.
# MAGIC 
# MAGIC <br clear="all"/>
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC For movie recommendations, we start with a matrix whose entries are movie ratings by users (shown in red in the diagram below).  Each column represents a user (shown in green) and each row represents a particular movie (shown in blue).
# MAGIC 
# MAGIC Since not all users have rated all movies, we do not know all of the entries in this matrix, which is precisely why we need collaborative filtering.  For each user, we have ratings for only a subset of the movies.  With collaborative filtering, the idea is to approximate the ratings matrix by factorizing it as the product of two matrices: one that describes properties of each user (shown in green), and one that describes properties of each movie (shown in blue).
# MAGIC 
# MAGIC <img alt="factorization" src="http://spark-mooc.github.io/web-assets/images/matrix_factorization.png" style="width: 885px"/>
# MAGIC <br clear="all"/>
# MAGIC 
# MAGIC We want to select these two matrices such that the error for the users/movie pairs where we know the correct ratings is minimized.  The [Alternating Least Squares][als] algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the movies such that the error is minimized.  Then, it holds the movies matrix constrant and optimizes the value of the user's matrix.  This alternation between which matrix to optimize is the reason for the "alternating" in the name.
# MAGIC 
# MAGIC This optimization is what's being shown on the right in the image above.  Given a fixed set of user factors (i.e., values in the users matrix), we use the known ratings to find the best values for the movie factors using the optimization written at the bottom of the figure.  Then we "alternate" and pick the best user factors given fixed movie factors.
# MAGIC 
# MAGIC For a simple example of what the users and movies matrices might look like, check out the [videos from Lecture 8][videos] or the [slides from Lecture 8][slides]
# MAGIC [videos]: https://courses.edx.org/courses/BerkeleyX/CS100.1x/1T2015/courseware/00eb8b17939b4889a41a6d8d2f35db83/3bd3bba368be4102b40780550d3d8da6/
# MAGIC [slides]: https://courses.edx.org/c4x/BerkeleyX/CS110x/asset/Lecture2.pdf
# MAGIC [als]: https://en.wikiversity.org/wiki/Least-Squares_Method
# MAGIC [mllib]: http://spark.apache.org/docs/1.6.2/mllib-guide.html
# MAGIC [collab]: https://en.wikipedia.org/?title=Collaborative_filtering
# MAGIC [collab2]: http://recommender-systems.org/collaborative-filtering/

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2a) Creating a Training Set
# MAGIC 
# MAGIC Before we jump into using machine learning, we need to break up the `ratings_df` dataset into three pieces:
# MAGIC * A training set (DataFrame), which we will use to train models
# MAGIC * A validation set (DataFrame), which we will use to choose the best model
# MAGIC * A test set (DataFrame), which we will use for our experiments
# MAGIC 
# MAGIC To randomly split the dataset into the multiple groups, we can use the pySpark [randomSplit()](http://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit) transformation. `randomSplit()` takes a set of splits and and seed and returns multiple DataFrames.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.

# We'll hold out 80% for training, 20% of our data for validation, and leave 20% for testing
seed = 1800009193L
(split_60_df, split_a_20_df, split_b_20_df) = <FILL_IN>

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

# COMMAND ----------

# TEST
Test.assertEquals(training_df.count(), 12001389, "Incorrect training_df count. Expected 12001389")
Test.assertEquals(validation_df.count(), 4003694, "Incorrect validation_df count. Expected 4003694")
Test.assertEquals(test_df.count(), 3995180, "Incorrect test_df count. Expected 3995180")

Test.assertEquals(training_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 5952) & (ratings_df.rating == 5.0)).count(), 1)
Test.assertEquals(training_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 1193) & (ratings_df.rating == 3.5)).count(), 1)
Test.assertEquals(training_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 1196) & (ratings_df.rating == 4.5)).count(), 1)

Test.assertEquals(validation_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 296) & (ratings_df.rating == 4.0)).count(), 1)
Test.assertEquals(validation_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 32) & (ratings_df.rating == 3.5)).count(), 1)
Test.assertEquals(validation_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 6888) & (ratings_df.rating == 3.0)).count(), 1)

Test.assertEquals(test_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 4993) & (ratings_df.rating == 5.0)).count(), 1)
Test.assertEquals(test_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 4128) & (ratings_df.rating == 4.0)).count(), 1)
Test.assertEquals(test_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 4915) & (ratings_df.rating == 3.0)).count(), 1)


# COMMAND ----------

# MAGIC %md
# MAGIC After splitting the dataset, your training set has about 12 million entries and the validation and test sets each have about 4 million entries. (The exact number of entries in each dataset varies slightly due to the random nature of the `randomSplit()` transformation.)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2b) Alternating Least Squares
# MAGIC 
# MAGIC In this part, we will use the Apache Spark ML Pipeline MLlib implementation of Alternating Least Squares, [ALS](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS). ALS takes a training dataset (RDD) and several parameters that control the model creation process. To determine the best values for the parameters, we will use ALS to train several models, and then we will select the best model and use the parameters from that model in the rest of this lab exercise.
# MAGIC 
# MAGIC The process we will use for determining the best model is as follows:
# MAGIC 1. Pick a set of model parameters. The most important parameter to is the *rank*, which is the number of rows in the Users matrix (green in the diagram above) or the number of columns in the Movies matrix (blue in the diagram above). In general, a lower rank will mean higher error on the training dataset, but a high rank may lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting).  We will train models with ranks of 4, 8, and 12 using the `training_df` dataset.
# MAGIC 
# MAGIC 2. Set the appropriate parameters on the `ALS` object:
# MAGIC     * The "User" column will be set to the values in our `userId` DataFrame column.
# MAGIC     * The "Item" column will be set to the values in our `movieId` DataFrame column.
# MAGIC     * The "Rating" column will be set to the values in our `rating` DataFrame column.
# MAGIC     * We'll using a regularization parameter of 0.1.
# MAGIC 
# MAGIC    **Note**: Read the documentation for the [ALS](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS) class **carefully**. It will help you accomplish this step.
# MAGIC 3. Have the ALS output transformation?i.e., the result of [ALS.fit()](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS.fit)?produce a _new_ column
# MAGIC    called "prediction" that contains the predicted value.
# MAGIC 
# MAGIC 4. Create multiple models using [ALS.fit()](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS.fit), one for each of our rank values. We'll fit
# MAGIC    against the training data set (`training_df`).
# MAGIC 
# MAGIC 5. For each model, we'll run a prediction against our validation data set (`validation_df`) and check the error.
# MAGIC 
# MAGIC 6. We'll keep the model with the best error rate.
# MAGIC 
# MAGIC #### Why are we doing are own cross-validation?
# MAGIC 
# MAGIC A challenge for collaborative filtering is how to provide ratings a new user?a user who has not provided *any* ratings at all. Some recommendation systems choose to provide new users with a default ratings (e.g., an average value across all ratings), while others choose to provide no ratings for new users. Spark?s ALS algorithm yields a NaN (?Not a Number?) value when asked to provide a rating for a new user.
# MAGIC 
# MAGIC Using the ML Pipeline?s [CrossValidator](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) with ALS is thus problematic, because cross validation involves dividing the training data into a set of folds (e.g., three sets) and then using those folds for testing and evaluating the parameters during the parameter grid search process. It is likely that some of the folds will contain users that are not in the other folds?and, as a result, ALS produces NaN values for those new users. When the CrossValidator uses the Evaluator (RMSE) to compute an error metric, the RMSE algorithm will return NaN. This will make *all* of the parameters in the parameter grid appear to be equally good (or bad).
# MAGIC 
# MAGIC You can read the discussion on (Spark JIRA 14489)[https://issues.apache.org/jira/browse/SPARK-14489] about this issue. There are proposed workarounds of having ALS provide default values or having RMSE drop NaN values. Both introduce potential issues. We?ve chosen to have RMSE drop NaN values. While this does not solve the underlying issue of ALS not predicting a value for a new user, it does provide some evaluation value. We manually implement the parameter grid search process using a for loop (below) and remove the NaN values before using RMSE.
# MAGIC 
# MAGIC For a production application, you would want to consider the tradeoffs in how to handle new users.
# MAGIC 
# MAGIC **Note**: This cell will likely a couple of minutes to run.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# This step is broken in ML Pipelines: https://issues.apache.org/jira/browse/SPARK-14489
from pyspark.ml.recommendation import ALS

# Let's initialize our ALS learner
als = ALS()

# Now we set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setRegParam(0.1)\
   .<FILL_IN>

# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12]
errors = [0, 0, 0]
models = [0, 0, 0]
err = 0
min_error = float('inf')
best_rank = -1
for rank in ranks:
  # Set the rank here:
  als.<FILL_IN>
  # Create the model with these parameters.
  model = als.fit(training_df)
  # Run the model to create a prediction. Predict against the validation_df.
  predict_df = model.<FILL_IN>

  # Remove NaN values from prediction (due to SPARK-14489)
  predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

  # Run the previously created RMSE evaluator, reg_eval, on the predictedRatingsDF DataFrame
  error = reg_eval.<FILL_IN>
  errors[err] = error
  models[err] = model
  print 'For rank %s the RMSE is %s' % (rank, error)
  if error < min_error:
    min_error = error
    best_rank = err
  err += 1

als.setRank(ranks[best_rank])
print 'The best model was trained with rank %s' % ranks[best_rank]
myModel = models[best_rank]

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2c) Testing Your Model
# MAGIC 
# MAGIC So far, we used the `training_df` and `validation_df` datasets to select the best model.  Since we used these two datasets to determine what model is best, we cannot use them to test how good the model is; otherwise, we would be very vulnerable to [overfitting](https://en.wikipedia.org/wiki/Overfitting).  To decide how good our model is, we need to use the `test_df` dataset.  We will use the `best_rank` you determined in part (2b) to create a model for predicting the ratings for the test dataset and then we will compute the RMSE.
# MAGIC 
# MAGIC The steps you should perform are:
# MAGIC * Run a prediction, using `my_model` as created above, on the test dataset (`test_df`), producing a new `predict_df` DataFrame.
# MAGIC * Filter out unwanted NaN values (necessary because of [a bug in Spark](https://issues.apache.org/jira/browse/SPARK-14489)). We've supplied this piece of code for you.
# MAGIC * Use the previously created RMSE evaluator, `reg_eval` to evaluate the filtered DataFrame.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code
# In ML Pipelines, this next step has a bug that produces unwanted NaN values. We
# have to filter them out. See https://issues.apache.org/jira/browse/SPARK-14489
predict_df = my_model.<FILL_IN>

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predictDF.prediction != float('nan'))

# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = <FILL_IN>

print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

# COMMAND ----------

# TEST
Test.assertTrue(abs(test_RMSE - 0.809624038485) < tolerance, 'incorrect test_RMSE: {0:.11f}'.format(test_RMSE))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2d) Comparing Your Model
# MAGIC 
# MAGIC Looking at the RMSE for the results predicted by the model versus the values in the test set is one way to evalute the quality of our model. Another way to evaluate the model is to evaluate the error from a test set where every rating is the average rating for the training set.
# MAGIC 
# MAGIC The steps you should perform are:
# MAGIC * Use the `training_df` to compute the average rating across all movies in that training dataset.
# MAGIC * Use the average rating that you just determined and the `test_df` to create a DataFrame (`test_for_avg_df`) with a `prediction` column containing the average rating. **HINT**: You'll want to use the `lit()` function,
# MAGIC   from `pyspark.sql.functions`, available here as `F.lit()`.
# MAGIC * Use our previously created `reg_eval` object to evaluate the `test_for_avg_df` and calculate the RMSE.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
# Compute the average rating
avg_rating_df = <FILL_IN>

# Extract the average rating value. (This is row 0, column 0.)
training_avg_rating = avg_rating_df.collect()[0][0]

print('The average rating for movies in the training set is {0}'.format(training_avg_rating))

# Add a column with the average rating
test_for_avg_df = training_df.withColumn('prediction', <FILL_IN>)

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = <FILL_IN>

print("The RMSE on the average set is {0}".format(test_avg_RMSE))

# COMMAND ----------

# TEST Comparing Your Model (2e)
Test.assertTrue(abs(training_avg_rating - 3.52547984237) < 0.000001,
                'incorrect training_avg_rating (expected 3.52547984237): {0:.11f}'.format(training_avg_rating))
Test.assertTrue(abs(test_avg_RMSE - 1.0519743756) < 0.000001,
                'incorrect test_avg_RMSE (expected 1.0519743756): {0:.11f}'.format(test_avg_RMSE))

# COMMAND ----------

# MAGIC %md
# MAGIC You now have code to predict how users will rate movies!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Predictions for Yourself
# MAGIC The ultimate goal of this lab exercise is to predict what movies to recommend to yourself.  In order to do that, you will first need to add ratings for yourself to the `ratingsRDD` dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC **(3a) Your Movie Ratings**
# MAGIC 
# MAGIC To help you provide ratings for yourself, we have included the following code to list the names and movie IDs of the 50 highest-rated movies from `movies_with_500_ratings_or_more` which we created in part 1 the lab.

# COMMAND ----------

print 'Most rated movies:'
print '(average rating, movie name, number of reviews, movie ID)'
display(movies_with_500_ratings_or_more.orderBy(movies_with_500_ratings_or_more['average'].desc()).take(50))

# COMMAND ----------

# MAGIC %md
# MAGIC The user ID 0 is unassigned, so we will use it for your ratings. We set the variable `my_user_ID` to 0 for you. Next, create a new DataFrame called `my_ratings_df`, with your ratings for at least 10 movie ratings. Each entry should be formatted as `(my_user_id, movieID, rating)`.  As in the original dataset, ratings should be between 1 and 5 (inclusive). If you have not seen at least 10 of these movies, you can increase the parameter passed to `take()` in the above cell until there are 10 movies that you have seen (or you can also guess what your rating would be for movies you have not seen).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql import Row
my_user_id = 0

# Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the movie ID.
my_rated_movies = [
     <FILL IN>
     # The format of each line is (my_user_id, movie ID, your rating)
     # For example, to give the movie "Star Wars: Episode IV - A New Hope (1977)" a five rating, you would add the following line:
     #   (my_user_id, 260, 5),
]

rows = [Row(userId=r[0], movieId=r[1], rating=float(r[2])) for r in my_rated_movies]
my_ratings_df = sqlContext.createDataFrame(rows)
print 'My movie ratings:'
display(my_ratings_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3b) Add Your Movies to Training Dataset
# MAGIC 
# MAGIC Now that you have ratings for yourself, you need to add your ratings to the `training` dataset so that the model you train will incorporate your preferences.  Spark's [union()](http://spark.apache.org/docs/latest/api/python/pyspark.rdd.RDD-class.html#union) transformation combines two RDDs; use `union()` to create a new training dataset that includes your ratings and the data in the original training dataset.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
training_with_my_ratings_df = <FILL IN>

print ('The training dataset now has %s more entries than the original training dataset' %
       (training_with_my_ratings_df.count() - training_df.count()))
assert (training_with_my_ratings_df.count() - training_df.count()) == my_ratings_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3c) Train a Model with Your Ratings
# MAGIC 
# MAGIC Now, train a model with your ratings added and the parameters you used in in part (2b) and (2c). Mke sure you include **all** of the parameters.
# MAGIC 
# MAGIC **Note**: This cell will take about 30 seconds to run.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

# Reset the parameters for the ALS object.
als.setPredictionCol("prediction")\
   .setMaxIter(5)\
   .setSeed(seed)\
   .<FILL_IN>

# Create the model with these parameters.
my_model = als.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3d) Check RMSE for the New Model with Your Ratings
# MAGIC 
# MAGIC Compute the RMSE for this new model on the test set.
# MAGIC * Run your model (the one you just trained) against the test data set in `test_df`.
# MAGIC * Then, use our previously-computed `reg_eval` object to compute the RMSE of your ratings.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
my_predict_df = my_ratings_model.<FILL IN>

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_my_ratings_df = my_predict_df.filter(my_predict_df.prediction != float('nan'))

# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_my_ratings_df DataFrame
test_RMSE_my_ratings = <FILL IN>
print('The model had a RMSE on the test set of {0}'.format(test_RMSE_my_ratings))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3e) Predict Your Ratings
# MAGIC 
# MAGIC So far, we have only computed the error of the model.  Next, let's predict what ratings you would give to the movies that you did not already provide ratings for.
# MAGIC 
# MAGIC The steps you should perform are:
# MAGIC * Filter out the movies you already rated manually. (Use the `my_rated_movie_ids` variable.) Put the results in a new `not_rated_df`.
# MAGIC 
# MAGIC    **Hint**: The [Column.isin()](http://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.Column.isin)
# MAGIC    method, as well as the `~` ("not") DataFrame logical operator, may come in handy here. Here's an example of using `isin()`:
# MAGIC 
# MAGIC ```
# MAGIC     > df1 = sqlContext.createDataFrame([("Jim", 10), ("Julie", 9), ("Abdul", 20), ("Mireille", 19)], ["name", "age"])
# MAGIC     > df1.show()
# MAGIC     +--------+---+
# MAGIC     |    name|age|
# MAGIC     +--------+---+
# MAGIC     |     Jim| 10|
# MAGIC     |   Julie|  9|
# MAGIC     |   Abdul| 20|
# MAGIC     |Mireille| 19|
# MAGIC     +--------+---+
# MAGIC 
# MAGIC     > names_to_delete = ("Julie", "Abdul") # this is just a Python tuple
# MAGIC     > df2 = df1.filter(~ df1["name"].isin(names_to_delete)) # "NOT IN"
# MAGIC     > df2.show()
# MAGIC     +--------+---+
# MAGIC     |    name|age|
# MAGIC     +--------+---+
# MAGIC     |     Jim| 10|
# MAGIC     |   Julie|  9|
# MAGIC     +--------+---+
# MAGIC ```
# MAGIC 
# MAGIC * Transform `not_rated_df` into `my_unrated_movies_df` by:
# MAGIC     - renaming the "ID" column to "movieId"
# MAGIC     - adding a "userId" column with the value contained in the `my_user_id` variable defined above.
# MAGIC 
# MAGIC * Create a `predicted_ratings_df` DataFrame by applying `my_ratings_model` to `my_unrated_movies_df`.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code

# Create a list of my rated movie IDs
my_rated_movie_ids = [x[1] for x in my_rated_movies]

# Filter out the movies I already rated.
not_rated_df = movies_df.<FILL_IN>

# Rename the "ID" column to be "movieId", and add a column with my_user_id as "userId".
my_unrated_movies_df = not_rated_df.<FILL_IN>

# Use my_rating_model to predict ratings for the movies that I did not manually rate.
predicted_ratings_df = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3f) Predict Your Ratings
# MAGIC 
# MAGIC We have our predicted ratings. Now we can print out the 25 movies with the highest predicted ratings.
# MAGIC 
# MAGIC The steps you should perform are:
# MAGIC * Join your `predicted_ratings_df` DataFrame with the `movie_names_with_avg_ratings_df` DataFrame to obtain the titles of the movies.
# MAGIC * Sort the resulting DataFrame (`predicted_with_counts_df`) by predicted rating (highest ratings first), and remove any ratings with a count of 75 or less.
# MAGIC * Print the top 25 movies that remain.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code

predicted_with_counts_df = <FILL_IN>
predicted_highest_rated_movies_df = predicted_with_counts_df.<FILL_IN>

print ('My 25 highest rated movies as predicted (for movies with more than 75 reviews):')
predicted_highest_rated_movies_df.<FILL_IN>

# COMMAND ----------


