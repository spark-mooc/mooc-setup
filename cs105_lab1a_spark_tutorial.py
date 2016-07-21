# Databricks notebook source exported at Fri, 24 Jun 2016 23:04:35 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC # **Spark Tutorial: Learning Apache Spark**
# MAGIC 
# MAGIC This tutorial will teach you how to use [Apache Spark](http://spark.apache.org/), a framework for large-scale data processing, within a notebook. Many traditional frameworks were designed to be run on a single computer.  However, many datasets today are too large to be stored on a single computer, and even when a dataset can be stored on one computer (such as the datasets in this tutorial), the dataset can often be processed much more quickly using multiple computers.
# MAGIC 
# MAGIC Spark has efficient implementations of a number of transformations and actions that can be composed together to perform data processing and analysis.  Spark excels at distributing these operations across a cluster while abstracting away many of the underlying implementation details.  Spark has been designed with a focus on scalability and efficiency.  With Spark you can begin developing your solution on your laptop, using a small dataset, and then use that same code to process terabytes or even petabytes across a distributed cluster.
# MAGIC 
# MAGIC **During this tutorial we will cover:**
# MAGIC 
# MAGIC * *Part 1:* Basic notebook usage and [Python](https://docs.python.org/2/) integration
# MAGIC * *Part 2:* An introduction to using [Apache Spark](https://spark.apache.org/) with the [PySpark SQL API](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark-sql-module) running in a notebook
# MAGIC * *Part 3:* Using DataFrames and chaining together transformations and actions
# MAGIC * *Part 4*: Python Lambda functions and User Defined Functions
# MAGIC * *Part 5:* Additional DataFrame actions
# MAGIC * *Part 6:* Additional DataFrame transformations
# MAGIC * *Part 7:* Caching DataFrames and storage options
# MAGIC * *Part 8:* Debugging Spark applications and lazy evaluation
# MAGIC 
# MAGIC The following transformations will be covered:
# MAGIC * `select()`, `filter()`, `distinct()`, `dropDuplicates()`, `orderBy()`, `groupBy()`
# MAGIC 
# MAGIC The following actions will be covered:
# MAGIC * `first()`, `take()`, `count()`, `collect()`, `show()`
# MAGIC 
# MAGIC Also covered:
# MAGIC * `cache()`, `unpersist()`
# MAGIC 
# MAGIC Note that, for reference, you can look up the details of these methods in the [Spark's PySpark SQL API](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark-sql-module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Part 1: Basic notebook usage and [Python](https://docs.python.org/2/) integration **

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1a) Notebook usage
# MAGIC 
# MAGIC A notebook is comprised of a linear sequence of cells.  These cells can contain either markdown or code, but we won't mix both in one cell.  When a markdown cell is executed it renders formatted text, images, and links just like HTML in a normal webpage.  The text you are reading right now is part of a markdown cell.  Python code cells allow you to execute arbitrary Python commands just like in any Python shell. Place your cursor inside the cell below, and press "Shift" + "Enter" to execute the code and advance to the next cell.  You can also press "Ctrl" + "Enter" to execute the code and remain in the cell.  These commands work the same in both markdown and code cells.

# COMMAND ----------

# This is a Python cell. You can run normal Python code here...
print 'The sum of 1 and 1 is {0}'.format(1+1)

# COMMAND ----------

# Here is another Python cell, this time with a variable (x) declaration and an if statement:
x = 42
if x > 40:
    print 'The sum of 1 and 2 is {0}'.format(1+2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1b) Notebook state
# MAGIC 
# MAGIC As you work through a notebook it is important that you run all of the code cells.  The notebook is stateful, which means that variables and their values are retained until the notebook is detached (in Databricks) or the kernel is restarted (in Jupyter notebooks).  If you do not run all of the code cells as you proceed through the notebook, your variables will not be properly initialized and later code might fail.  You will also need to rerun any cells that you have modified in order for the changes to be available to other cells.

# COMMAND ----------

# This cell relies on x being defined already.
# If we didn't run the cells from part (1a) this code would fail.
print x * 2

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1c) Library imports
# MAGIC 
# MAGIC We can import standard Python libraries ([modules](https://docs.python.org/2/tutorial/modules.html)) the usual way.  An `import` statement will import the specified module.  In this tutorial and future labs, we will provide any imports that are necessary.

# COMMAND ----------

# Import the regular expression library
import re
m = re.search('(?<=abc)def', 'abcdef')
m.group(0)

# COMMAND ----------

# Import the datetime library
import datetime
print 'This was last run on: {0}'.format(datetime.datetime.now())


# COMMAND ----------

# MAGIC %md
# MAGIC ##  **Part 2: An introduction to using [Apache Spark](https://spark.apache.org/) with the [PySpark SQL API](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark-sql-module) running in a notebook**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark Context
# MAGIC 
# MAGIC In Spark, communication occurs between a driver and executors.  The driver has Spark jobs that it needs to run and these jobs are split into tasks that are submitted to the executors for completion.  The results from these tasks are delivered back to the driver.
# MAGIC 
# MAGIC In part 1, we saw that normal Python code can be executed via cells. When using Databricks this code gets executed in the Spark driver's Java Virtual Machine (JVM) and not in an executor's JVM, and when using an Jupyter notebook it is executed within the kernel associated with the notebook. Since no Spark functionality is actually being used, no tasks are launched on the executors.
# MAGIC 
# MAGIC In order to use Spark and its DataFrame API we will need to use a `SQLContext`.  When running Spark, you start a new Spark application by creating a [SparkContext](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext). You can then create a [SQLContext](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SQLContext) from the `SparkContext`. When the `SparkContext` is created, it asks the master for some cores to use to do work.  The master sets these cores aside just for you; they won't be used for other applications. When using Databricks, both a `SparkContext` and a `SQLContext` are created for you automatically. `sc` is your `SparkContext`, and `sqlContext` is your `SQLContext`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2a) Example Cluster
# MAGIC The diagram shows an example cluster, where the slots allocated for an application are outlined in purple. (Note: We're using the term _slots_ here to indicate threads available to perform parallel work for Spark.
# MAGIC Spark documentation often refers to these threads as _cores_, which is a confusing term, as the number of slots available on a particular machine does not necessarily have any relationship to the number of physical CPU
# MAGIC cores on that machine.)
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs105x/diagram-2a.png" style="height: 800px;float: right"/>
# MAGIC 
# MAGIC You can view the details of your Spark application in the Spark web UI.  The web UI is accessible in Databricks by going to "Clusters" and then clicking on the "Spark UI" link for your cluster.  In the web UI, under the "Jobs" tab, you can see a list of jobs that have been scheduled or run.  It's likely there isn't any thing interesting here yet because we haven't run any jobs, but we'll return to this page later.
# MAGIC 
# MAGIC At a high level, every Spark application consists of a driver program that launches various parallel operations on executor Java Virtual Machines (JVMs) running either in a cluster or locally on the same machine. In Databricks, "Databricks Shell" is the driver program.  When running locally, `pyspark` is the driver program. In all cases, this driver program contains the main loop for the program and creates distributed datasets on the cluster, then applies operations (transformations & actions) to those datasets.
# MAGIC Driver programs access Spark through a SparkContext object, which represents a connection to a computing cluster. A Spark SQL context object (`sqlContext`) is the main entry point for Spark DataFrame and SQL functionality. A `SQLContext` can be used to create DataFrames, which allows you to direct the operations on your data.
# MAGIC 
# MAGIC Try printing out `sqlContext` to see its type.

# COMMAND ----------

# Display the type of the Spark sqlContext
type(sqlContext)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the type is `HiveContext`. This means we're working with a version of Spark that has Hive support. Compiling Spark with Hive support is a good idea, even if you don't have a Hive metastore. As the
# MAGIC [Spark Programming Guide](http://spark.apache.org/docs/latest/sql-programming-guide.html#starting-point-sqlcontext) states, a `HiveContext` "provides a superset of the functionality provided by the basic `SQLContext`. Additional features include the ability to write queries using the more complete HiveQL parser, access to Hive UDFs [user-defined functions], and the ability to read data from Hive tables. To use a `HiveContext`, you do not need to have an existing Hive setup, and all of the data sources available to a `SQLContext` are still available."

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2b) SparkContext attributes
# MAGIC 
# MAGIC You can use Python's [dir()](https://docs.python.org/2/library/functions.html?highlight=dir#dir) function to get a list of all the attributes (including methods) accessible through the `sqlContext` object.

# COMMAND ----------

# List sqlContext's attributes
dir(sqlContext)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2c) Getting help
# MAGIC 
# MAGIC Alternatively, you can use Python's [help()](https://docs.python.org/2/library/functions.html?highlight=help#help) function to get an easier to read list of all the attributes, including examples, that the `sqlContext` object has.

# COMMAND ----------

# Use help to obtain more detailed information
help(sqlContext)

# COMMAND ----------

# MAGIC %md
# MAGIC Outside of `pyspark` or a notebook, `SQLContext` is created from the lower-level `SparkContext`, which is usually used to create Resilient Distributed Datasets (RDDs). An RDD is the way Spark actually represents data internally; DataFrames are actually implemented in terms of RDDs.
# MAGIC 
# MAGIC While you can interact directly with RDDs, DataFrames are preferred. They're generally faster, and they perform the same no matter what language (Python, R, Scala or Java) you use with Spark.
# MAGIC 
# MAGIC In this course, we'll be using DataFrames, so we won't be interacting directly with the Spark Context object very much. However, it's worth knowing that inside `pyspark` or a notebook, you already have an existing `SparkContext` in the `sc` variable. One simple thing we can do with `sc` is check the version of Spark we're using:

# COMMAND ----------

# After reading the help we've decided we want to use sc.version to see what version of Spark we are running
sc.version

# COMMAND ----------

# Help can be used on any Python object
help(map)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Part 3: Using DataFrames and chaining together transformations and actions**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Working with your first DataFrames
# MAGIC 
# MAGIC In Spark, we first create a base [DataFrame](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame). We can then apply one or more transformations to that base DataFrame. *A DataFrame is immutable, so once it is created, it cannot be changed.* As a result, each transformation creates a new DataFrame. Finally, we can apply one or more actions to the DataFrames.
# MAGIC 
# MAGIC > Note that Spark uses lazy evaluation, so transformations are not actually executed until an action occurs.
# MAGIC 
# MAGIC We will perform several exercises to obtain a better understanding of DataFrames:
# MAGIC * Create a Python collection of 10,000 integers
# MAGIC * Create a Spark DataFrame from that collection
# MAGIC * Subtract one from each value using `map`
# MAGIC * Perform action `collect` to view results
# MAGIC * Perform action `count` to view counts
# MAGIC * Apply transformation `filter` and view results with `collect`
# MAGIC * Learn about lambda functions
# MAGIC * Explore how lazy evaluation works and the debugging challenges that it introduces
# MAGIC 
# MAGIC A DataFrame consists of a series of `Row` objects; each `Row` object has a set of named columns. You can think of a DataFrame as modeling a table, though the data source being processed does not have to be a table.
# MAGIC 
# MAGIC More formally, a DataFrame must have a _schema_, which means it must consist of columns, each of which has a _name_ and a _type_. Some data sources have schemas built into them. Examples include RDBMS databases, Parquet files, and NoSQL databases like Cassandra. Other data sources don't have computer-readable schemas, but you can often apply a schema programmatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3a) Create a Python collection of 10,000 people
# MAGIC 
# MAGIC We will use a third-party Python testing library called [fake-factory](https://pypi.python.org/pypi/fake-factory/0.5.3) to create a collection of fake person records.

# COMMAND ----------

from faker import Factory
fake = Factory.create()
fake.seed(4321)

# COMMAND ----------

# MAGIC %md
# MAGIC We're going to use this factory to create a collection of randomly generated people records. In the next section, we'll turn that collection into a DataFrame. We'll use a Python tuple to help us define the Spark DataFrame schema. There are other ways to define schemas, though; see
# MAGIC the Spark Programming Guide's discussion of [schema inference](http://spark.apache.org/docs/latest/sql-programming-guide.html#inferring-the-schema-using-reflection) for more information. (For instance,
# MAGIC we could also use a Python `namedtuple` or a Spark `Row` object.)

# COMMAND ----------

# Each entry consists of last_name, first_name, ssn, job, and age (at least 1)
from pyspark.sql import Row
def fake_entry():
  name = fake.name().split()
  return (name[1], name[0], fake.ssn(), fake.job(), abs(2016 - fake.date_time().year) + 1)

# COMMAND ----------

# Create a helper function to call a function repeatedly
def repeat(times, func, *args, **kwargs):
    for _ in xrange(times):
        yield func(*args, **kwargs)

# COMMAND ----------

data = list(repeat(10000, fake_entry))

# COMMAND ----------

# MAGIC %md
# MAGIC `data` is just a normal Python list, containing Python tuples objects. Let's look at the first item in the list:

# COMMAND ----------

data[0]

# COMMAND ----------

# MAGIC %md
# MAGIC We can check the size of the list using the Python `len()` function.

# COMMAND ----------

len(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3b) Distributed data and using a collection to create a DataFrame
# MAGIC 
# MAGIC In Spark, datasets are represented as a list of entries, where the list is broken up into many different partitions that are each stored on a different machine.  Each partition holds a unique subset of the entries in the list.  Spark calls datasets that it stores "Resilient Distributed Datasets" (RDDs). Even DataFrames are ultimately represented as RDDs, with additional meta-data.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3b.png" style="width: 900px; float: right; margin: 5px"/>
# MAGIC 
# MAGIC One of the defining features of Spark, compared to other data analytics frameworks (e.g., Hadoop), is that it stores data in memory rather than on disk.  This allows Spark applications to run much more quickly, because they are not slowed down by needing to read data from disk.
# MAGIC The figure to the right illustrates how Spark breaks a list of data entries into partitions that are each stored in memory on a worker.
# MAGIC 
# MAGIC 
# MAGIC To create the DataFrame, we'll use `sqlContext.createDataFrame()`, and we'll pass our array of data in as an argument to that function. Spark will create a new set of input data based on data that is passed in.  A DataFrame requires a _schema_, which is a list of columns, where each column has a name and a type. Our list of data has elements with types (mostly strings, but one integer). We'll supply the rest of the schema and the column names as the second argument to `createDataFrame()`.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's view the help for `createDataFrame()`.

# COMMAND ----------

help(sqlContext.createDataFrame)

# COMMAND ----------

dataDF = sqlContext.createDataFrame(data, ('last_name', 'first_name', 'ssn', 'occupation', 'age'))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see what type `sqlContext.createDataFrame()` returned.

# COMMAND ----------

print 'type of dataDF: {0}'.format(type(dataDF))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the DataFrame's schema and some of its rows.

# COMMAND ----------

dataDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC We can register the newly created DataFrame as a named table, using the `registerDataFrameAsTable()` method.

# COMMAND ----------

sqlContext.registerDataFrameAsTable(dataDF, 'dataframe')

# COMMAND ----------

# MAGIC %md
# MAGIC What methods can we call on this DataFrame?

# COMMAND ----------

help(dataDF)

# COMMAND ----------

# MAGIC %md
# MAGIC How many partitions will the DataFrame be split into?

# COMMAND ----------

dataDF.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### A note about DataFrames and queries
# MAGIC 
# MAGIC When you use DataFrames or Spark SQL, you are building up a _query plan_. Each transformation you apply to a DataFrame adds some information to the query plan. When you finally call an action, which triggers execution of your Spark job, several things happen:
# MAGIC 
# MAGIC 1. Spark's Catalyst optimizer analyzes the query plan (called an _unoptimized logical query plan_) and attempts to optimize it. Optimizations include (but aren't limited to) rearranging and combining `filter()` operations for efficiency, converting `Decimal` operations to more efficient long integer operations, and pushing some operations down into the data source (e.g., a `filter()` operation might be translated to a SQL `WHERE` clause, if the data source is a traditional SQL RDBMS). The result of this optimization phase is an _optimized logical plan_.
# MAGIC 2. Once Catalyst has an optimized logical plan, it then constructs multiple _physical_ plans from it. Specifically, it implements the query in terms of lower level Spark RDD operations.
# MAGIC 3. Catalyst chooses which physical plan to use via _cost optimization_. That is, it determines which physical plan is the most efficient (or least expensive), and uses that one.
# MAGIC 4. Finally, once the physical RDD execution plan is established, Spark actually executes the job.
# MAGIC 
# MAGIC You can examine the query plan using the `explain()` function on a DataFrame. By default, `explain()` only shows you the final physical plan; however, if you pass it an argument of `True`, it will show you all phases.
# MAGIC 
# MAGIC (If you want to take a deeper dive into how Catalyst optimizes DataFrame queries, this blog post, while a little old, is an excellent overview: [Deep Dive into Spark SQL's Catalyst Optimizer](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html).)
# MAGIC 
# MAGIC Let's add a couple transformations to our DataFrame and look at the query plan on the resulting transformed DataFrame. Don't be too concerned if it looks like gibberish. As you gain more experience with Apache Spark, you'll begin to be able to use `explain()` to help you understand more about your DataFrame operations.

# COMMAND ----------

newDF = dataDF.distinct().select('*')
newDF.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3c): Subtract one from each value using _select_
# MAGIC 
# MAGIC So far, we've created a distributed DataFrame that is split into many partitions, where each partition is stored on a single machine in our cluster.  Let's look at what happens when we do a basic operation on the dataset.  Many useful data analysis operations can be specified as "do something to each item in the dataset".  These data-parallel operations are convenient because each item in the dataset can be processed individually: the operation on one entry doesn't effect the operations on any of the other entries.  Therefore, Spark can parallelize the operation.
# MAGIC 
# MAGIC One of the most common DataFrame operations is `select()`, and it works more or less like a SQL `SELECT` statement: You can select specific columns from the DataFrame, and you can even use `select()` to create _new_ columns with values that are derived from existing column values. We can use `select()` to create a new column that decrements the value of the existing `age` column.
# MAGIC 
# MAGIC `select()` is a _transformation_. It returns a new DataFrame that captures both the previous DataFrame and the operation to add to the query (`select`, in this case). But it does *not* actually execute anything on the cluster. When transforming DataFrames, we are building up a _query plan_. That query plan will be optimized, implemented (in terms of RDDs), and executed by Spark _only_ when we call an action.

# COMMAND ----------

# Transform dataDF through a select transformation and rename the newly created '(age -1)' column to 'age'
# Because select is a transformation and Spark uses lazy evaluation, no jobs, stages,
# or tasks will be launched when we run this code.
subDF = dataDF.select('last_name', 'first_name', 'ssn', 'occupation', (dataDF.age - 1).alias('age'))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the query plan.

# COMMAND ----------

subDF.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3d) Use _collect_ to view results
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3d.png" style="height:700px;float:right"/>
# MAGIC 
# MAGIC To see a list of elements decremented by one, we need to create a new list on the driver from the the data distributed in the executor nodes.  To do this we can call the `collect()` method on our DataFrame.  `collect()` is often used after transformations to ensure that we are only returning a *small* amount of data to the driver.  This is done because the data returned to the driver must fit into the driver's available memory.  If not, the driver will crash.
# MAGIC 
# MAGIC The `collect()` method is the first action operation that we have encountered.  Action operations cause Spark to perform the (lazy) transformation operations that are required to compute the values returned by the action.  In our example, this means that tasks will now be launched to perform the `createDataFrame`, `select`, and `collect` operations.
# MAGIC 
# MAGIC In the diagram, the dataset is broken into four partitions, so four `collect()` tasks are launched. Each task collects the entries in its partition and sends the result to the driver, which creates a list of the values, as shown in the figure below.
# MAGIC 
# MAGIC Now let's run `collect()` on `subDF`.

# COMMAND ----------

# Let's collect the data
results = subDF.collect()
print results

# COMMAND ----------

# MAGIC %md
# MAGIC A better way to visualize the data is to use the `show()` method. If you don't tell `show()` how many rows to display, it displays 20 rows.

# COMMAND ----------

subDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC If you'd prefer that `show()` not truncate the data, you can tell it not to:

# COMMAND ----------

subDF.show(n=30, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC In Databricks, there's an even nicer way to look at the values in a DataFrame: The `display()` helper function.

# COMMAND ----------

display(subDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3e) Use _count_ to get total
# MAGIC 
# MAGIC One of the most basic jobs that we can run is the `count()` job which will count the number of elements in a DataFrame, using the `count()` action. Since `select()` creates a new DataFrame with the same number of elements as the starting DataFrame, we expect that applying `count()` to each DataFrame will return the same result.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3e.png" style="height:700px;float:right"/>
# MAGIC 
# MAGIC Note that because `count()` is an action operation, if we had not already performed an action with `collect()`, then Spark would now perform the transformation operations when we executed `count()`.
# MAGIC 
# MAGIC Each task counts the entries in its partition and sends the result to your SparkContext, which adds up all of the counts. The figure on the right shows what would happen if we ran `count()` on a small example dataset with just four partitions.

# COMMAND ----------

print dataDF.count()
print subDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3f) Apply transformation _filter_ and view results with _collect_
# MAGIC 
# MAGIC Next, we'll create a new DataFrame that only contains the people whose ages are less than 10. To do this, we'll use the `filter()` transformation. (You can also use `where()`, an alias for `filter()`, if you prefer something more SQL-like). The `filter()` method is a transformation operation that creates a new DataFrame from the input DataFrame, keeping only values that match the filter expression.
# MAGIC 
# MAGIC The figure shows how this might work on the small four-partition dataset.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3f.png" style="height:700px;float:right"/>
# MAGIC 
# MAGIC To view the filtered list of elements less than 10, we need to create a new list on the driver from the distributed data on the executor nodes.  We use the `collect()` method to return a list that contains all of the elements in this filtered DataFrame to the driver program.

# COMMAND ----------

filteredDF = subDF.filter(subDF.age < 10)
filteredDF.show(truncate=False)
filteredDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC (These are some _seriously_ precocious children...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Python Lambda functions and User Defined Functions
# MAGIC 
# MAGIC Python supports the use of small one-line anonymous functions that are not bound to a name at runtime.
# MAGIC 
# MAGIC `lambda` functions, borrowed from LISP, can be used wherever function objects are required. They are syntactically restricted to a single expression. Remember that `lambda` functions are a matter of style and using them is never required - semantically, they are just syntactic sugar for a normal function definition. You can always define a separate normal function instead, but using a `lambda` function is an equivalent and more compact form of coding. Ideally you should consider using `lambda` functions where you want to encapsulate non-reusable code without littering your code with one-line functions.
# MAGIC 
# MAGIC Here, instead of defining a separate function for the `filter()` transformation, we will use an inline `lambda()` function and we will register that lambda as a Spark _User Defined Function_ (UDF). A UDF is a special wrapper around a function, allowing the function to be used in a DataFrame query.

# COMMAND ----------

from pyspark.sql.types import BooleanType
less_ten = udf(lambda s: s < 10, BooleanType())
lambdaDF = subDF.filter(less_ten(subDF.age))
lambdaDF.show()
lambdaDF.count()

# COMMAND ----------

# Let's collect the even values less than 10
even = udf(lambda s: s % 2 == 0, BooleanType())
evenDF = lambdaDF.filter(even(lambdaDF.age))
evenDF.show()
evenDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Additional DataFrame actions
# MAGIC 
# MAGIC Let's investigate some additional actions:
# MAGIC 
# MAGIC * [first()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.first)
# MAGIC * [take()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.take)
# MAGIC 
# MAGIC One useful thing to do when we have a new dataset is to look at the first few entries to obtain a rough idea of what information is available.  In Spark, we can do that using actions like `first()`, `take()`, and `show()`. Note that for the `first()` and `take()` actions, the elements that are returned depend on how the DataFrame is *partitioned*.
# MAGIC 
# MAGIC Instead of using the `collect()` action, we can use the `take(n)` action to return the first _n_ elements of the DataFrame. The `first()` action returns the first element of a DataFrame, and is equivalent to `take(1)[0]`.

# COMMAND ----------

print "first: {0}\n".format(filteredDF.first())

print "Four of them: {0}\n".format(filteredDF.take(4))

# COMMAND ----------

# MAGIC %md
# MAGIC This looks better:

# COMMAND ----------

display(filteredDF.take(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Additional DataFrame transformations

# COMMAND ----------

# MAGIC %md
# MAGIC ### (6a) _orderBy_
# MAGIC 
# MAGIC [`orderBy()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.distinct) allows you to sort a DataFrame by one or more columns, producing a new DataFrame.
# MAGIC 
# MAGIC For example, let's get the first five oldest people in the original (unfiltered) DataFrame. We can use the `orderBy()` transformation. `orderBy` takes one or more columns, either as _names_ (strings) or as `Column` objects. To get a `Column` object, we use one of two notations on the DataFrame:
# MAGIC 
# MAGIC * Pandas-style notation: `filteredDF.age`
# MAGIC * Subscript notation: `filteredDF['age']`
# MAGIC 
# MAGIC Both of those syntaxes return a `Column`, which has additional methods like `desc()` (for sorting in descending order) or `asc()` (for sorting in ascending order, which is the default).
# MAGIC 
# MAGIC Here are some examples:
# MAGIC 
# MAGIC ```
# MAGIC dataDF.orderBy(dataDF['age'])  # sort by age in ascending order; returns a new DataFrame
# MAGIC dataDF.orderBy(dataDF.last_name.desc()) # sort by last name in descending order
# MAGIC ```

# COMMAND ----------

# Get the five oldest people in the list. To do that, sort by age in descending order.
display(dataDF.orderBy(dataDF.age.desc()).take(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's reverse the sort order. Since ascending sort is the default, we can actually use a `Column` object expression or a simple string, in this case. The `desc()` and `asc()` methods are only defined on `Column`. Something like `orderBy('age'.desc())` would not work, because there's no `desc()` method on Python string objects. That's why we needed the column expression. But if we're just using the defaults, we can pass a string column name into `orderBy()`. This is sometimes easier to read.

# COMMAND ----------

display(dataDF.orderBy('age').take(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (6b) _distinct_ and _dropDuplicates_
# MAGIC 
# MAGIC [`distinct()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.distinct) filters out duplicate rows, and it considers all columns. Since our data is completely randomly generated (by `fake-factory`), it's extremely unlikely that there are any duplicate rows:

# COMMAND ----------

print dataDF.count()
print dataDF.distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC To demonstrate `distinct()`, let's create a quick throwaway dataset.

# COMMAND ----------

tempDF = sqlContext.createDataFrame([("Joe", 1), ("Joe", 1), ("Anna", 15), ("Anna", 12), ("Ravi", 5)], ('name', 'score'))

# COMMAND ----------

tempDF.show()

# COMMAND ----------

tempDF.distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that one of the ("Joe", 1) rows was deleted, but both rows with name "Anna" were kept, because all columns in a row must match another row for it to be considered a duplicate.

# COMMAND ----------

# MAGIC %md
# MAGIC [`dropDuplicates()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropDuplicates) is like `distinct()`, except that it allows us to specify the columns to compare. For instance, we can use it to drop all rows where the first name and last name duplicates (ignoring the occupation and age columns).

# COMMAND ----------

print dataDF.count()
print dataDF.dropDuplicates(['first_name', 'last_name']).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (6c) _drop_
# MAGIC 
# MAGIC [`drop()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.drop) is like the opposite of `select()`: Instead of selecting specific columns from a DataFrame, it drops a specifed column from a DataFrame.
# MAGIC 
# MAGIC Here's a simple use case: Suppose you're reading from a 1,000-column CSV file, and you have to get rid of five of the columns. Instead of selecting 995 of the columns, it's easier just to drop the five you don't want.

# COMMAND ----------

dataDF.drop('occupation').drop('age').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (6d) _groupBy_
# MAGIC 
# MAGIC [`groupBy()`]((http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.groupBy) is one of the most powerful transformations. It allows you to perform aggregations on a DataFrame.
# MAGIC 
# MAGIC Unlike other DataFrame transformations, `groupBy()` does _not_ return a DataFrame. Instead, it returns a special [GroupedData](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData) object that contains various aggregation functions.
# MAGIC 
# MAGIC The most commonly used aggregation function is [count()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.count),
# MAGIC but there are others (like [sum()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.sum), [max()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.max), and [avg()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.avg).
# MAGIC 
# MAGIC These aggregation functions typically create a new column and return a new DataFrame.

# COMMAND ----------

dataDF.groupBy('occupation').count().show(truncate=False)

# COMMAND ----------

dataDF.groupBy().avg('age').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use `groupBy()` to do aother useful aggregations:

# COMMAND ----------

print "Maximum age: {0}".format(dataDF.groupBy().max('age').first()[0])
print "Minimum age: {0}".format(dataDF.groupBy().min('age').first()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### (6e) _sample_ (optional)
# MAGIC 
# MAGIC When analyzing data, the [`sample()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample) transformation is often quite useful. It returns a new DataFrame with a random sample of elements from the dataset.  It takes in a `withReplacement` argument, which specifies whether it is okay to randomly pick the same item multiple times from the parent DataFrame (so when `withReplacement=True`, you can get the same item back multiple times). It takes in a `fraction` parameter, which specifies the fraction elements in the dataset you want to return. (So a `fraction` value of `0.20` returns 20% of the elements in the DataFrame.) It also takes an optional `seed` parameter that allows you to specify a seed value for the random number generator, so that reproducible results can be obtained.

# COMMAND ----------

sampledDF = dataDF.sample(withReplacement=False, fraction=0.10)
print sampledDF.count()
sampledDF.show()

# COMMAND ----------

print dataDF.sample(withReplacement=False, fraction=0.05).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 7: Caching DataFrames and storage options

# COMMAND ----------

# MAGIC %md
# MAGIC ### (7a) Caching DataFrames
# MAGIC 
# MAGIC For efficiency Spark keeps your DataFrames in memory. (More formally, it keeps the _RDDs_ that implement your DataFrames in memory.) By keeping the contents in memory, Spark can quickly access the data. However, memory is limited, so if you try to keep too many partitions in memory, Spark will automatically delete partitions from memory to make space for new ones. If you later refer to one of the deleted partitions, Spark will automatically recreate it for you, but that takes time.
# MAGIC 
# MAGIC So, if you plan to use a DataFrame more than once, then you should tell Spark to cache it. You can use the `cache()` operation to keep the DataFrame in memory. However, you must still trigger an action on the DataFrame, such as `collect()` or `count()` before the caching will occur. In other words, `cache()` is lazy: It merely tells Spark that the DataFrame should be cached _when the data is materialized_. You have to run an action to materialize the data; the DataFrame will be cached as a side effect. The next time you use the DataFrame, Spark will use the cached data, rather than recomputing the DataFrame from the original data.
# MAGIC 
# MAGIC You can see your cached DataFrame in the "Storage" section of the Spark web UI. If you click on the name value, you can see more information about where the the DataFrame is stored.

# COMMAND ----------

# Cache the DataFrame
filteredDF.cache()
# Trigger an action
print filteredDF.count()
# Check if it is cached
print filteredDF.is_cached

# COMMAND ----------

# MAGIC %md
# MAGIC ### (7b) Unpersist and storage options
# MAGIC 
# MAGIC Spark automatically manages the partitions cached in memory. If it has more partitions than available memory, by default, it will evict older partitions to make room for new ones. For efficiency, once you are finished using cached DataFrame, you can optionally tell Spark to stop caching it in memory by using the DataFrame's `unpersist()` method to inform Spark that you no longer need the cached data.
# MAGIC 
# MAGIC ** Advanced: ** Spark provides many more options for managing how DataFrames cached. For instance, you can tell Spark to spill cached partitions to disk when it runs out of memory, instead of simply throwing old ones away. You can explore the API for DataFrame's [persist()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.persist) operation using Python's [help()](https://docs.python.org/2/library/functions.html?highlight=help#help) command.  The `persist()` operation, optionally, takes a pySpark [StorageLevel](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.StorageLevel) object.

# COMMAND ----------

# If we are done with the DataFrame we can unpersist it so that its memory can be reclaimed
filteredDF.unpersist()
# Check if it is cached
print filteredDF.is_cached

# COMMAND ----------

# MAGIC %md
# MAGIC ## ** Part 8: Debugging Spark applications and lazy evaluation **

# COMMAND ----------

# MAGIC %md
# MAGIC ### How Python is Executed in Spark
# MAGIC 
# MAGIC Internally, Spark executes using a Java Virtual Machine (JVM). pySpark runs Python code in a JVM using [Py4J](http://py4j.sourceforge.net). Py4J enables Python programs running in a Python interpreter to dynamically access Java objects in a Java Virtual Machine. Methods are called as if the Java objects resided in the Python interpreter and Java collections can be accessed through standard Python collection methods. Py4J also enables Java programs to call back Python objects.
# MAGIC 
# MAGIC Because pySpark uses Py4J, coding errors often result in a complicated, confusing stack trace that can be difficult to understand. In the following section, we'll explore how to understand stack traces.

# COMMAND ----------

# MAGIC %md
# MAGIC ### (8a) Challenges with lazy evaluation using transformations and actions
# MAGIC 
# MAGIC Spark's use of lazy evaluation can make debugging more difficult because code is not always executed immediately. To see an example of how this can happen, let's first define a broken filter function.
# MAGIC Next we perform a `filter()` operation using the broken filtering function.  No error will occur at this point due to Spark's use of lazy evaluation.
# MAGIC 
# MAGIC The `filter()` method will not be executed *until* an action operation is invoked on the DataFrame.  We will perform an action by using the `count()` method to return a list that contains all of the elements in this DataFrame.

# COMMAND ----------

def brokenTen(value):
    """Incorrect implementation of the ten function.

    Note:
        The `if` statement checks an undefined variable `val` instead of `value`.

    Args:
        value (int): A number.

    Returns:
        bool: Whether `value` is less than ten.

    Raises:
        NameError: The function references `val`, which is not available in the local or global
            namespace, so a `NameError` is raised.
    """
    if (val < 10):
        return True
    else:
        return False

btUDF = udf(brokenTen)
brokenDF = subDF.filter(btUDF(subDF.age) == True)

# COMMAND ----------

# Now we'll see the error
# Click on the `+` button to expand the error and scroll through the message.
brokenDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (8b) Finding the bug
# MAGIC 
# MAGIC When the `filter()` method is executed, Spark calls the UDF. Since our UDF has an error in the underlying filtering function `brokenTen()`, an error occurs.
# MAGIC 
# MAGIC Scroll through the output "Py4JJavaError     Traceback (most recent call last)" part of the cell and first you will see that the line that generated the error is the `count()` method line. There is *nothing wrong with this line*. However, it is an action and that caused other methods to be executed. Continue scrolling through the Traceback and you will see the following error line:
# MAGIC 
# MAGIC `NameError: global name 'val' is not defined`
# MAGIC 
# MAGIC Looking at this error line, we can see that we used the wrong variable name in our filtering function `brokenTen()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### (8c) Moving toward expert style
# MAGIC 
# MAGIC As you are learning Spark, I recommend that you write your code in the form:
# MAGIC ```
# MAGIC     df2 = df1.transformation1()
# MAGIC     df2.action1()
# MAGIC     df3 = df2.transformation2()
# MAGIC     df3.action2()
# MAGIC ```
# MAGIC Using this style will make debugging your code much easier as it makes errors easier to localize - errors in your transformations will occur when the next action is executed.
# MAGIC 
# MAGIC Once you become more experienced with Spark, you can write your code with the form: `df.transformation1().transformation2().action()`
# MAGIC 
# MAGIC We can also use `lambda()` functions instead of separately defined functions when their use improves readability and conciseness.

# COMMAND ----------

# Cleaner code through lambda use
myUDF = udf(lambda v: v < 10)
subDF.filter(myUDF(subDF.age) == True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (8d) Readability and code style
# MAGIC 
# MAGIC To make the expert coding style more readable, enclose the statement in parentheses and put each method, transformation, or action on a separate line.

# COMMAND ----------

# Final version
from pyspark.sql.functions import *
(dataDF
 .filter(dataDF.age > 20)
 .select(concat(dataDF.first_name, lit(' '), dataDF.last_name), dataDF.occupation)
 .show(truncate=False)
 )

# COMMAND ----------

