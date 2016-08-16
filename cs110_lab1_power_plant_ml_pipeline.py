# Databricks notebook source exported at Tue, 16 Aug 2016 13:14:29 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC #Power Plant Machine Learning Pipeline Application
# MAGIC This notebook is an end-to-end exercise of performing Extract-Transform-Load and Exploratory Data Analysis on a real-world dataset, and then applying several different machine learning algorithms to solve a supervised regression problem on the dataset.
# MAGIC 
# MAGIC ** This notebook covers: **
# MAGIC * *Part 1: Business Understanding*
# MAGIC * *Part 2: Load Your Data*
# MAGIC * *Part 3: Explore Your Data*
# MAGIC * *Part 4: Visualize Your Data*
# MAGIC * *Part 5: Data Preparation*
# MAGIC * *Part 6: Data Modeling*
# MAGIC * *Part 7: Tuning and Evaluation*
# MAGIC 
# MAGIC *Our goal is to accurately predict power output given a set of environmental readings from various sensors in a natural gas-fired power generation plant.*
# MAGIC 
# MAGIC 
# MAGIC ** Background **
# MAGIC 
# MAGIC Power generation is a complex process, and understanding and predicting power output is an important element in managing a plant and its connection to the power grid. The operators of a regional power grid create predictions of power demand based on historical information and environmental factors (e.g., temperature). They then compare the predictions against available resources (e.g., coal, natural gas, nuclear, solar, wind, hydro power plants). Power generation technologies such as solar and wind are highly dependent on environmental conditions, and all generation technologies are subject to planned and unplanned maintenance.
# MAGIC 
# MAGIC Here is an real-world example of predicted demand (on two time scales), actual demand, and available resources from the California power grid: http://www.caiso.com/Pages/TodaysOutlook.aspx
# MAGIC 
# MAGIC ![](http://content.caiso.com/outlook/SP/ems_small.gif)
# MAGIC 
# MAGIC The challenge for a power grid operator is how to handle a shortfall in available resources versus actual demand. There are three solutions to  a power shortfall: build more base load power plants (this process can take many years to decades of planning and construction), buy and import power from other regional power grids (this choice can be very expensive and is limited by the power transmission interconnects between grids and the excess power available from other grids), or turn on small [Peaker or Peaking Power Plants](https://en.wikipedia.org/wiki/Peaking_power_plant). Because grid operators need to respond quickly to a power shortfall to avoid a power outage, grid operators rely on a combination of the last two choices. In this exercise, we'll focus on the last choice.
# MAGIC 
# MAGIC ** The Business Problem **
# MAGIC 
# MAGIC Because they supply power only occasionally, the power supplied by a peaker power plant commands a much higher price per kilowatt hour than power from a power grid's base power plants. A peaker plant may operate many hours a day, or it may operate only a few hours per year, depending on the condition of the region's electrical grid. Because of the cost of building an efficient power plant, if a peaker plant is only going to be run for a short or highly variable time it does not make economic sense to make it as efficient as a base load power plant. In addition, the equipment and fuels used in base load plants are often unsuitable for use in peaker plants because the fluctuating conditions would severely strain the equipment.
# MAGIC 
# MAGIC The power output of a peaker power plant varies depending on environmental conditions, so the business problem is _predicting the power output of a peaker power plant as a function of the environmental conditions_ -- since this would enable the grid operator to make economic tradeoffs about the number of peaker plants to turn on (or whether to buy expensive power from another grid).
# MAGIC 
# MAGIC Given this business problem, we need to first perform Exploratory Data Analysis to understand the data and then translate the business problem (predicting power output as a function of envionmental conditions) into a Machine Learning task.  In this instance, the ML task is regression since the label (or target) we are trying to predict is numeric. We will use an [Apache Spark ML Pipeline](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark-ml-package) to perform the regression.
# MAGIC 
# MAGIC The real-world data we are using in this notebook consists of 9,568 data points, each with 4 environmental attributes collected from a Combined Cycle Power Plant over 6 years (2006-2011), and is provided by the University of California, Irvine at [UCI Machine Learning Repository Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant). You can find more details about the dataset on the UCI page, including the following background publications:
# MAGIC * Pinar Tüfekci, [Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods](http://www.journals.elsevier.com/international-journal-of-electrical-power-and-energy-systems/), International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615.
# MAGIC * Heysem Kaya, Pinar Tüfekci and Fikret S. Gürgen: [Local and Global Learning Methods for Predicting Power of a Combined Gas & Steam Turbine](http://www.cmpe.boun.edu.tr/~kaya/kaya2012gasturbine.pdf), Proceedings of the International Conference on Emerging Trends in Computer and Electronics Engineering ICETCEE 2012, pp. 13-18 (Mar. 2012, Dubai).
# MAGIC 
# MAGIC **To Do**: Read the documentation and examples for [Spark Machine Learning Pipeline](https://spark.apache.org/docs/1.6.2/ml-guide.html#main-concepts-in-pipelines).

# COMMAND ----------

labVersion = 'cs110x-power-plant-1.0.0'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Business Understanding
# MAGIC The first step in any machine learning task is to understand the business need.
# MAGIC 
# MAGIC As described in the overview we are trying to predict power output given a set of readings from various sensors in a gas-fired power generation plant.
# MAGIC 
# MAGIC The problem is a regression problem since the label (or target) we are trying to predict is numeric.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Extract-Transform-Load (ETL) Your Data
# MAGIC 
# MAGIC Now that we understand what we are trying to do, the first step is to load our data into a format we can query and use.  This is known as ETL or "Extract-Transform-Load".  We will load our file from Amazon S3.
# MAGIC 
# MAGIC Note: Alternatively we could upload our data using "Databricks Menu > Tables > Create Table", assuming we had the raw files on our local computer.
# MAGIC 
# MAGIC Our data is available on Amazon s3 at the following path:
# MAGIC 
# MAGIC ```
# MAGIC dbfs:/databricks-datasets/power-plant/data
# MAGIC ```
# MAGIC 
# MAGIC **To Do:** Let's start by printing a sample of the data.
# MAGIC 
# MAGIC We'll use the built-in Databricks functions for exploring the Databricks filesystem (DBFS)
# MAGIC 
# MAGIC Use `display(dbutils.fs.ls("/databricks-datasets/power-plant/data"))` to list the files in the directory

# COMMAND ----------

display(dbutils.fs.ls("/databricks-datasets/power-plant/data"))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, use the `dbutils.fs.head` command to look at the first 65,536 bytes of the first file in the directory.
# MAGIC 
# MAGIC Use `print dbutils.fs.head("/databricks-datasets/power-plant/data/Sheet1.tsv")` to list the files in the directory

# COMMAND ----------

print dbutils.fs.head("/databricks-datasets/power-plant/data/Sheet1.tsv")

# COMMAND ----------

# MAGIC %md
# MAGIC `dbutils.fs` has its own help facility, which we can use to see the various available functions.

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exercise 2(a)
# MAGIC 
# MAGIC Now, let's use PySpark instead to print the first 5 lines of the data.
# MAGIC 
# MAGIC *Hint*: First create an RDD from the data by using [`sc.textFile("dbfs:/databricks-datasets/power-plant/data")`](https://spark.apache.org/docs/1.6.2/api/python/pyspark.html#pyspark.SparkContext.textFile) to read the data into an RDD.
# MAGIC 
# MAGIC *Hint*: Then figure out how to use the RDD [`take()`](https://spark.apache.org/docs/1.6.2/api/python/pyspark.html#pyspark.RDD.take) method to extract the first 5 lines of the RDD and print each line.

# COMMAND ----------

# TODO: Load the data and print the first five lines.
rawTextRdd = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC From our initial exploration of a sample of the data, we can make several observations for the ETL process:
# MAGIC   - The data is a set of .tsv (Tab Seperated Values) files (i.e., each row of the data is separated using tabs)
# MAGIC   - There is a header row, which is the name of the columns
# MAGIC   - It looks like the type of the data in each column is consistent (i.e., each column is of type double)
# MAGIC 
# MAGIC Our schema definition from UCI appears below:
# MAGIC - AT = Atmospheric Temperature in C
# MAGIC - V = Exhaust Vacuum Speed
# MAGIC - AP = Atmospheric Pressure
# MAGIC - RH = Relative Humidity
# MAGIC - PE = Power Output.  This is the value we are trying to predict given the measurements above.
# MAGIC 
# MAGIC We are ready to create a DataFrame from the TSV data. Spark does not have a native method for performing this operation, however we can use [spark-csv](https://spark-packages.org/package/databricks/spark-csv), a third-party package from [SparkPackages](https://spark-packages.org/). The documentation and source code for [spark-csv](https://spark-packages.org/package/databricks/spark-csv) can be found on [GitHub](https://github.com/databricks/spark-csv). The Python API can be found [here](https://github.com/databricks/spark-csv#python-api).
# MAGIC 
# MAGIC (**Note**: In Spark 2.0, the CSV package is built into the DataFrame API.)
# MAGIC 
# MAGIC To use the [spark-csv](https://spark-packages.org/package/databricks/spark-csv) package, we use the [sqlContext.read.format()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.format) method to specify the input data source format: `'com.databricks.spark.csv'`
# MAGIC 
# MAGIC We can provide the [spark-csv](https://spark-packages.org/package/databricks/spark-csv) package with options using the [options()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.options) method. The available options are listed in the GitHub documentation [here](https://github.com/databricks/spark-csv#features).
# MAGIC 
# MAGIC We will use the following three options:
# MAGIC - `delimiter='\t'` because our data is tab delimited
# MAGIC - `header='true'` because our data has a header row
# MAGIC - `inferschema='true'` because we believe that all of the data is double values, so the package can dynamically infer the type of each column. *Note that this will require two pass over the data.*
# MAGIC 
# MAGIC 
# MAGIC The last component of creating the DataFrame is to specify the location of the data source using the [load()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.load) method: `"/databricks-datasets/power-plant/data"`
# MAGIC 
# MAGIC Putting everything together, we will use an operation of the following form:
# MAGIC 
# MAGIC   `sqlContext.read.format().options().load()`
# MAGIC 
# MAGIC ### Exercise 2(b)
# MAGIC 
# MAGIC **To Do:** Create a DataFrame from the data.
# MAGIC 
# MAGIC *Hint:* Use the above template and fill in each of the methods.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
powerPlantDF = sqlContext.read.format(<FILL_IN>).options(<FILL_IN>).load(<FILL_IN>)

# COMMAND ----------

# TEST
from databricks_test_helper import *
expected = set([(s, 'double') for s in ('AP', 'AT', 'PE', 'RH', 'V')])
Test.assertEquals(expected, set(powerPlantDF.dtypes), "Incorrect schema for powerPlantDF")


# COMMAND ----------

# MAGIC %md
# MAGIC Check the names and types of the columns using the [dtypes](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dtypes) method.

# COMMAND ----------

print powerPlantDF.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC We can examine the data using the display() method.

# COMMAND ----------

display(powerPlantDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 2: Alternative Method to Load your Data
# MAGIC Instead of having [spark-csv](https://spark-packages.org/package/databricks/spark-csv) infer the types of the columns, we can specify the schema as a [DataType](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.types.DataType), which is a list of [StructField](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.types.StructType).
# MAGIC 
# MAGIC You can find a list of types in the [pyspark.sql.types](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#module-pyspark.sql.types) module. For our data, we will use [DoubleType()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.types.DoubleType).
# MAGIC 
# MAGIC For example, to specify that a column's name and type, we use: `StructField(`_name_`,` _type_`, True)`. (The third parameter, `True`, signifies that the column is nullable.)
# MAGIC 
# MAGIC ### Exercise 2(c)
# MAGIC 
# MAGIC Create a custom schema for the power plant data.

# COMMAND ----------

# TO DO: Fill in the custom schema.
from pyspark.sql.types import *

# Custom Schema for Power Plant
customSchema = StructType([ \
    <FILL_IN>, \
    <FILL_IN>, \
    <FILL_IN>, \
    <FILL_IN>, \
    <FILL_IN> \
                          ])

# COMMAND ----------

# TEST
Test.assertEquals(set([f.name for f in customSchema.fields]), set(['AT', 'V', 'AP', 'RH', 'PE']), 'Incorrect column names in schema.')
Test.assertEquals(set([f.dataType for f in customSchema.fields]), set([DoubleType(), DoubleType(), DoubleType(), DoubleType(), DoubleType()]), 'Incorrect column types in schema.')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 2(d)
# MAGIC 
# MAGIC Now, let's use the schema to read the data. To do this, we will modify the earlier `sqlContext.read.format` step. We can specify the schema by:
# MAGIC - Adding `schema = customSchema` to the load method (use a comma and add it after the file name)
# MAGIC - Removing the `inferschema='true'`option because we are explicitly specifying the schema

# COMMAND ----------

# TODO: Use the schema you created above to load the data again.
altPowerPlantDF = sqlContext.read.format(<FILL_IN>).options(<FILL_IN>).load(<FILL_IN>)

# COMMAND ----------

# TEST
from databricks_test_helper import *
expected = set([(s, 'double') for s in ('AP', 'AT', 'PE', 'RH', 'V')])
Test.assertEquals(expected, set(altPowerPlantDF.dtypes), "Incorrect schema for powerPlantDF")


# COMMAND ----------

# MAGIC %md
# MAGIC Note that no Spark jobs are launched this time. That is because we specified the schema, so the [spark-csv](https://spark-packages.org/package/databricks/spark-csv) package does not have to read the data to infer the schema. We can use the [dtypes](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dtypes) method to examine the names and types of the columns. They should be identical to the names and types of the columns that were earlier inferred from the data.
# MAGIC 
# MAGIC When you run the following cell, data would not be read.

# COMMAND ----------

print altPowerPlantDF.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can examine the data using the display() method. *Note that this operation will cause the data to be read and the DataFrame will be created.*

# COMMAND ----------

display(altPowerPlantDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Explore Your Data
# MAGIC Now that your data is loaded, the next step is to explore it and perform some basic analysis and visualizations.
# MAGIC 
# MAGIC This is a step that you should always perform **before** trying to fit a model to the data, as this step will often lead to important insights about your data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC First, let's register our DataFrame as an SQL table named `power_plant`.  Because you may run this lab multiple times, we'll take the precaution of removing any existing tables first.
# MAGIC 
# MAGIC We can delete any existing `power_plant` SQL table using the SQL command: `DROP TABLE IF EXISTS power_plant` (we also need to to delete any Hive data associated with the table, which we can do with a Databricks file system operation).
# MAGIC 
# MAGIC Once any prior table is removed, we can register our DataFrame as a SQL table using [sqlContext.registerDataFrameAsTable()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.SQLContext.registerDataFrameAsTable).
# MAGIC 
# MAGIC ### 3(a)
# MAGIC 
# MAGIC **ToDo:** Execute the prepared code in the following cell.

# COMMAND ----------

sqlContext.sql("DROP TABLE IF EXISTS power_plant")
dbutils.fs.rm("dbfs:/user/hive/warehouse/power_plant", True)
sqlContext.registerDataFrameAsTable(powerPlantDF, "power_plant")

# COMMAND ----------

# MAGIC %md
# MAGIC Now that our DataFrame exists as a SQL table, we can explore it using SQL commands.
# MAGIC 
# MAGIC To execute SQL in a cell, we use the `%sql` operator. The following cell is an example of using SQL to query the rows of the SQL table.
# MAGIC 
# MAGIC **NOTE**: `%sql` is a Databricks-only command. It calls `sqlContext.sql()` and passes the results to the Databricks-only `display()` function. These two statements are equivalent:
# MAGIC 
# MAGIC `%sql SELECT * FROM power_plant`
# MAGIC 
# MAGIC `display(sqlContext.sql("SELECT * FROM power_plant"))`
# MAGIC 
# MAGIC ### 3(b)
# MAGIC 
# MAGIC **ToDo**: Execute the prepared code in the following cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- We can use %sql to query the rows
# MAGIC SELECT * FROM power_plant

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3(c)
# MAGIC 
# MAGIC Use the SQL `desc` command to describe the schema, by executing the following cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC desc power_plant

# COMMAND ----------

# MAGIC %md
# MAGIC **Schema Definition**
# MAGIC 
# MAGIC Once again, here's our schema definition:
# MAGIC 
# MAGIC - AT = Atmospheric Temperature in C
# MAGIC - V = Exhaust Vacuum Speed
# MAGIC - AP = Atmospheric Pressure
# MAGIC - RH = Relative Humidity
# MAGIC - PE = Power Output
# MAGIC 
# MAGIC PE is our label or target. This is the value we are trying to predict given the measurements.
# MAGIC 
# MAGIC *Reference [UCI Machine Learning Repository Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)*

# COMMAND ----------

# MAGIC %md
# MAGIC Let's perform some basic statistical analyses of all the columns.
# MAGIC 
# MAGIC We can get the DataFrame associated with a SQL table by using the [sqlContext.table()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.table) method and passing in the name of the SQL table. Then, we can use the DataFrame [describe()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.describe) method with no arguments to compute some basic statistics for each column like count, mean, max, min and standard deviation.

# COMMAND ----------

df = sqlContext.table("power_plant")
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 4: Visualize Your Data
# MAGIC 
# MAGIC To understand our data, we will look for correlations between features and the label.  This can be important when choosing a model.  E.g., if features and a label are linearly correlated, a linear model like Linear Regression can do well; if the relationship is very non-linear, more complex models such as Decision Trees can be better. We can use Databrick's built in visualization to view each of our predictors in relation to the label column as a scatter plot to see the correlation between the predictors and the label.
# MAGIC 
# MAGIC ### Exercise 4(a)
# MAGIC 
# MAGIC ** Add figures to the following: **
# MAGIC Let's see if there is a corellation between Temperature and Power Output. We can use a SQL query to create a new table consisting of only the Temperature (AT) and Power (PE) columns, and then use a scatter plot with Temperature on the X axis and Power on the Y axis to visualize the relationship (if any) between Temperature and Power.
# MAGIC 
# MAGIC 
# MAGIC Perform the following steps:
# MAGIC 
# MAGIC - Run the following cell
# MAGIC - Click on the drop down next to the "Bar chart" icon and select "Scatter" to turn the table into a Scatter plot
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/change-plot-type-scatter.png" style="border: 1px solid #999999"/>
# MAGIC 
# MAGIC - Click on "Plot Options..."
# MAGIC - In the Values box, click on "Temperature" and drag it before "Power", as shown below:
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/customize-plot-scatter.png" style="border: 1px solid #999999"/>
# MAGIC 
# MAGIC - Apply your changes by clicking the Apply button
# MAGIC - Increase the size of the graph by clicking and dragging the size control

# COMMAND ----------

# MAGIC %sql
# MAGIC select AT as Temperature, PE as Power from power_plant

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like there is strong linear correlation between Temperature and Power Output.
# MAGIC 
# MAGIC ** ASIDE: A quick physics lesson**: This correlation is to be expected as the second law of thermodynamics puts a fundamental limit on the [thermal efficiency](https://en.wikipedia.org/wiki/Thermal_efficiency) of all heat-based engines. The limiting factors are:
# MAGIC  - The temperature at which the heat enters the engine \\( T_{H} \\)
# MAGIC  - The temperature of the environment into which the engine exhausts its waste heat \\( T_C \\)
# MAGIC 
# MAGIC Our temperature measurements are the temperature of the environment. From [Carnot's theorem](https://en.wikipedia.org/wiki/Carnot%27s_theorem_%28thermodynamics%29), no heat engine working between these two temperatures can exceed the Carnot Cycle efficiency:
# MAGIC \\[ n_{th} \le 1 - \frac{T_C}{T_H}  \\]
# MAGIC 
# MAGIC Note that as the environmental temperature increases, the efficiency decreases -- _this is the effect that we see in the above graph._

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exercise 4(b)
# MAGIC 
# MAGIC Use SQL to create a scatter plot of Power(PE) as a function of ExhaustVacuum (V).
# MAGIC Name the y-axis "Power" and the x-axis "ExhaustVacuum"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TO DO: Replace <FILL_IN> with the appropriate SQL command.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's continue exploring the relationships (if any) between the variables and Power Output.
# MAGIC 
# MAGIC ### Exercise 4(c)
# MAGIC 
# MAGIC Use SQL to create a scatter plot of Power(PE) as a function of Pressure (AP).
# MAGIC Name the y-axis "Power" and the x-axis "Pressure"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TO DO: Replace <FILL_IN> with the appropriate SQL command.
# MAGIC <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exercise 4(d)
# MAGIC 
# MAGIC Use SQL to create a scatter plot of Power(PE) as a function of Humidity (RH).
# MAGIC Name the y-axis "Power" and the x-axis "Humidity"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TO DO: Replace <FILL_IN> with the appropriate SQL command.
# MAGIC <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 5: Data Preparation
# MAGIC 
# MAGIC The next step is to prepare the data for machine learning. Since all of this data is numeric and consistent this is a simple and straightforward task.
# MAGIC 
# MAGIC The goal is to use machine learning to determine a function that yields the output power as a function of a set of predictor features. The first step in building our ML pipeline is to convert the predictor features from DataFrame columns to Feature Vectors using the [pyspark.ml.feature.VectorAssembler()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler) method.
# MAGIC 
# MAGIC The VectorAssembler is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. VectorAssembler takes a list of input column names (each is a string) and the name of the output column (as a string).
# MAGIC 
# MAGIC ### Exercise 5(a)
# MAGIC 
# MAGIC - Read the Spark documentation and useage examples for [VectorAssembler](https://spark.apache.org/docs/1.6.2/ml-features.html#vectorassembler)
# MAGIC - Convert the `power_plant` SQL table into a DataFrame named `dataset`
# MAGIC - Set the vectorizer's input columns to a list of the four columns of the input DataFrame: `["AT", "V", "AP", "RH"]`
# MAGIC - Set the vectorizer's output column name to `"features"`

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code
from pyspark.ml.feature import VectorAssembler

datasetDF = <FILL_IN>

vectorizer = VectorAssembler()
vectorizer.setInputCols(<FILL_IN>)
vectorizer.setOutputCol(<FILL_IN>)

# COMMAND ----------

# TEST
Test.assertEquals(set(vectorizer.getInputCols()), {"AT", "V", "AP", "RH"}, "Incorrect vectorizer input columns")
Test.assertEquals(vectorizer.getOutputCol(), "features", "Incorrect vectorizer output column")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 6: Data Modeling
# MAGIC Now let's model our data to predict what the power output will be given a set of sensor readings
# MAGIC 
# MAGIC Our first model will be based on simple linear regression since we saw some linear patterns in our data based on the scatter plots during the exploration stage.
# MAGIC 
# MAGIC We need a way of evaluating how well our linear regression model predicts power output as a function of input parameters. We can do this by splitting up our initial data set into a _Training Set_ used to train our model and a _Test Set_ used to evaluate the model's performance in giving predictions. We can use a DataFrame's [randomSplit()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit) method to split our dataset. The method takes a list of weights and an optional random seed. The seed is used to initialize the random number generator used by the splitting function.
# MAGIC 
# MAGIC ### Exercise 6(a)
# MAGIC 
# MAGIC Use the [randomSplit()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit) method to divide up `datasetDF` into a trainingSetDF (80% of the input DataFrame) and a testSetDF (20% of the input DataFrame), and for reproducibility, use the seed 1800009193L. Then cache each DataFrame in memory to maximize performance.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
# We'll hold out 20% of our data for testing and leave 80% for training
seed = 1800009193L
(split20DF, split80DF) = datasetDF.<FILL_IN>

# Let's cache these datasets for performance
testSetDF = <FILL_IN>
trainingSetDF = <FILL_IN>

# COMMAND ----------

# TEST
Test.assertEquals(trainingSetDF.count(), 38243, "Incorrect size for training data set")
Test.assertEquals(testSetDF.count(), 9597, "Incorrect size for test data set")

# COMMAND ----------

# MAGIC %md
# MAGIC Next we'll create a Linear Regression Model and use the built in help to identify how to train it. See API details for [Linear Regression](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression) in the ML guide.
# MAGIC 
# MAGIC ### Exercise 6(b)
# MAGIC 
# MAGIC - Read the documentation and examples for [Linear Regression](https://spark.apache.org/docs/1.6.2/ml-classification-regression.html#linear-regression)
# MAGIC - Run the next cell

# COMMAND ----------

# ***** LINEAR REGRESSION MODEL ****

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

# Let's initialize our linear regression learner
lr = LinearRegression()

# We use explain params to dump the parameters we can use
print(lr.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC The cell below is based on the [Spark ML Pipeline API for Linear Regression](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression).
# MAGIC 
# MAGIC The first step is to set the parameters for the method:
# MAGIC - Set the name of the prediction column to "Predicted_PE"
# MAGIC - Set the name of the label column to "PE"
# MAGIC - Set the maximum number of iterations to 100
# MAGIC - Set the regularization parameter to 0.1
# MAGIC 
# MAGIC Next, we create the [ML Pipeline](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.Pipeline) and set the stages to the Vectorizer and Linear Regression learner we created earlier.
# MAGIC 
# MAGIC Finally, we create a model by training on `trainingSetDF`.
# MAGIC 
# MAGIC ### Exercise 6(c)
# MAGIC 
# MAGIC - Read the [Linear Regression](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression) documentation
# MAGIC - Run the next cell, and be sure you understand what's going on.

# COMMAND ----------

# Now we set the parameters for the method
lr.setPredictionCol("Predicted_PE")\
  .setLabelCol("PE")\
  .setMaxIter(100)\
  .setRegParam(0.1)


# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()

lrPipeline.setStages([vectorizer, lr])

# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSetDF)

# COMMAND ----------

# MAGIC %md
# MAGIC From the Wikipedia article on [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression):
# MAGIC > In statistics, linear regression is an approach for modeling the relationship between a scalar dependent variable \\( y \\) and one or more explanatory variables (or independent variables) denoted \\(X\\). In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models.
# MAGIC 
# MAGIC Linear regression has many practical uses. Most applications fall into one of the following two broad categories:
# MAGIC   - If the goal is prediction, or forecasting, or error reduction, linear regression can be used to fit a predictive model to an observed data set of \\(y\\) and \\(X\\) values. After developing such a model, if an additional value of \\(X\\) is then given without its accompanying value of \\(y\\), the fitted model can be used to make a prediction of the value of \\(y\\).
# MAGIC   - Given a variable \\(y\\) and a number of variables \\( X_1 \\), ..., \\( X_p \\) that may be related to \\(y\\), linear regression analysis can be applied to quantify the strength of the relationship between \\(y\\) and the \\( X_j\\), to assess which \\( X_j \\) may have no relationship with \\(y\\) at all, and to identify which subsets of the \\( X_j \\) contain redundant information about \\(y\\).
# MAGIC 
# MAGIC We are interested in both uses, as we would like to predict power output as a function of the input variables, and we would like to know which input variables are weakly or strongly correlated with power output.
# MAGIC 
# MAGIC Since Linear Regression is simply a Line of best fit over the data that minimizes the square of the error, given multiple input dimensions we can express each predictor as a line function of the form:
# MAGIC 
# MAGIC \\[ y = a + b x_1 + b x_2 + b x_i ... \\]
# MAGIC 
# MAGIC where \\(a\\) is the intercept and the \\(b\\) are the coefficients.
# MAGIC 
# MAGIC To express the coefficients of that line we can retrieve the Estimator stage from the PipelineModel and express the weights and the intercept for the function.
# MAGIC 
# MAGIC ### Exercise 6(d)
# MAGIC 
# MAGIC Run the next cell. Ensure that you understand what's going on.

# COMMAND ----------

# The intercept is as follows:
intercept = lrModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in datasetDF.columns if col != "PE"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

# Now let's sort the coefficients from greatest absolute weight most to the least absolute weight
coefficents.sort(key=lambda tup: abs(tup[0]), reverse=True)

equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

# Finally here is our equation
print("Linear Regression Equation: " + equation)


# COMMAND ----------

# MAGIC %md
# MAGIC Recall **Part 4: Visualize Your Data** when we visualized each predictor against Power Output using a Scatter Plot, does the final equation seems logical given those visualizations?
# MAGIC 
# MAGIC **ToDo**: Answer the quiz questions about correlations (on edX).
# MAGIC 
# MAGIC ### Exercise 6(e)
# MAGIC 
# MAGIC Now let's see what our predictions look like given this model. We apply our Linear Regression model to the 20% of the data that we split from the input dataset. The output of the model will be a predicted Power Output column named "Predicted_PE".
# MAGIC 
# MAGIC - Run the next cell
# MAGIC - Scroll through the resulting table and notice how the values in the Power Output (PE) column compare to the corresponding values in the predicted Power Output (Predicted_PE) column

# COMMAND ----------

# Apply our LR model to the test data and predict power output
predictionsAndLabelsDF = lrModel.transform(testSetDF).select("AT", "V", "AP", "RH", "PE", "Predicted_PE")

display(predictionsAndLabelsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC From a visual inspection of the predictions, we can see that they are close to the actual values.
# MAGIC 
# MAGIC However, we would like a scientific measure of how well the Linear Regression model is performing in accurately predicting values. To perform this measurement, we can use an evaluation metric such as [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) to validate our Linear Regression model.
# MAGIC 
# MAGIC RSME is defined as follows: \\( RMSE = \sqrt{\frac{\sum_{i = 1}^{n} (x_i - y_i)^2}{n}}\\) where \\(y_i\\) is the observed value and \\(x_i\\) is the predicted value
# MAGIC 
# MAGIC RMSE is a frequently used measure of the differences between values predicted by a model or an estimator and the values actually observed. The lower the RMSE, the better our model.
# MAGIC 
# MAGIC Spark ML Pipeline provides several regression analysis metrics, including [RegressionEvaluator()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator).
# MAGIC 
# MAGIC After we create an instance of [RegressionEvaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator), we set the label column name to "PE" and set the prediction column name to "Predicted_PE". We then invoke the evaluator on the predictions.
# MAGIC 
# MAGIC ### Exercise 6(f)
# MAGIC 
# MAGIC Run the next cell and ensure that you understand what's going on.

# COMMAND ----------

# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
regEval = RegressionEvaluator(predictionCol="Predicted_PE", labelCol="PE", metricName="rmse")

# Run the evaluator on the DataFrame
rmse = regEval.evaluate(predictionsAndLabelsDF)

print("Root Mean Squared Error: %.2f" % rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC Another useful statistical evaluation metric is the coefficient of determination, denoted \\(R^2\\) or \\(r^2\\) and pronounced "R squared". It is a number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable and it provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model. The coefficient of determination ranges from 0 to 1 (closer to 1), and the higher the value, the better our model.
# MAGIC 
# MAGIC To compute \\(r^2\\), we invoke the evaluator with  `regEval.metricName: "r2"`
# MAGIC 
# MAGIC ### Exercise 6(g)
# MAGIC 
# MAGIC Run the next cell and ensure that you understand what's going on.

# COMMAND ----------

# Now let's compute another evaluation metric for our test dataset
r2 = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

print("r2: {0:.2f}".format(r2))

# COMMAND ----------

# MAGIC %md
# MAGIC Generally, assuming a Gaussian distribution of errors, a good model will have 68% of predictions within 1 RMSE and 95% within 2 RMSE of the actual value (see http://statweb.stanford.edu/~susan/courses/s60/split/node60.html).
# MAGIC 
# MAGIC Let's examine the predictions and see if a RMSE of 4.59 meets this criteria.
# MAGIC 
# MAGIC We create a new DataFrame using [selectExpr()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.selectExpr) to project a set of SQL expressions, and register the DataFrame as a SQL table using [registerTempTable()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.registerTempTable).
# MAGIC 
# MAGIC ### Exercise 6(h)
# MAGIC 
# MAGIC Run the next cell and ensure that you understand what's going on.

# COMMAND ----------

# First we remove the table if it already exists
sqlContext.sql("DROP TABLE IF EXISTS Power_Plant_RMSE_Evaluation")
dbutils.fs.rm("dbfs:/user/hive/warehouse/Power_Plant_RMSE_Evaluation", True)

# Next we calculate the residual error and divide it by the RMSE
predictionsAndLabelsDF.selectExpr("PE", "Predicted_PE", "PE - Predicted_PE Residual_Error", "(PE - Predicted_PE) / {} Within_RSME".format(rmse)).registerTempTable("Power_Plant_RMSE_Evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC We can use SQL to explore the `Power_Plant_RMSE_Evaluation` table. First let's look at at the table using a SQL SELECT statement.
# MAGIC 
# MAGIC ### Exercise 6(i)
# MAGIC 
# MAGIC Run the next cell and ensure that you understand what's going on.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from Power_Plant_RMSE_Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can display the RMSE as a Histogram.
# MAGIC 
# MAGIC ### Exercise 6(j)
# MAGIC 
# MAGIC Perform the following steps:
# MAGIC 
# MAGIC - Run the following cell
# MAGIC - Click on the drop down next to the "Bar chart" icon a select "Histogram" to turn the table into a Histogram plot
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/change-plot-type-histogram.png" style="border: 1px solid #999999"/>
# MAGIC 
# MAGIC 
# MAGIC - Click on "Plot Options..."
# MAGIC - In the "All Fields:" box, click on "&lt;id&gt;" and drag it into the "Keys:" box
# MAGIC - Change the "Aggregation" to "COUNT"
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/customize-plot-histogram.png" style="border: 1px solid #999999"/>
# MAGIC 
# MAGIC - Apply your changes by clicking the Apply button
# MAGIC - Increase the size of the graph by clicking and dragging the size control
# MAGIC 
# MAGIC Notice that the histogram clearly shows that the RMSE is centered around 0 with the vast majority of the error within 2 RMSEs.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now we can display the RMSE as a Histogram
# MAGIC SELECT Within_RSME  from Power_Plant_RMSE_Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC Using a complex SQL SELECT statement, we can count the number of predictions within + or - 1.0 and + or - 2.0 and then display the results as a pie chart.
# MAGIC 
# MAGIC ### Exercise 6(k)
# MAGIC 
# MAGIC Perform the following steps:
# MAGIC 
# MAGIC   - Run the following cell
# MAGIC   - Click on the drop down next to the "Bar chart" icon a select "Pie" to turn the table into a Pie Chart plot
# MAGIC   - Increase the size of the graph by clicking and dragging the size control

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1
# MAGIC             when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3
# MAGIC        end RSME_Multiple, COUNT(*) AS count
# MAGIC FROM Power_Plant_RMSE_Evaluation
# MAGIC GROUP BY case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1  when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3 end

# COMMAND ----------

# MAGIC %md
# MAGIC From the pie chart, we can see that 68% of our test data predictions are within 1 RMSE of the actual values, and 97% (68% + 29%) of our test data predictions are within 2 RMSE. So the model is pretty decent. Let's see if we can tune the model to improve it further.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 7: Tuning and Evaluation
# MAGIC 
# MAGIC Now that we have a model with all of the data let's try to make a better model by tuning over several parameters. The process of tuning a model is known as [Model Selection](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#module-pyspark.ml.tuning) or [Hyperparameter Tuning](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#module-pyspark.ml.tuning), and Spark ML Pipeline makes the tuning process very simple and easy.
# MAGIC 
# MAGIC An important task in ML is model selection, or using data to find the best model or parameters for a given task. This is also called tuning. Tuning may be done for individual Estimators such as [LinearRegression](https://spark.apache.org/docs/1.6.2/ml-classification-regression.html#linear-regression), or for entire Pipelines which include multiple algorithms, featurization, and other steps. Users can tune an entire Pipeline at once, rather than tuning each element in the Pipeline separately.
# MAGIC 
# MAGIC Spark ML Pipeline supports model selection using tools such as [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator), which requires the following items:
# MAGIC   - [Estimator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.Estimator): algorithm or Pipeline to tune
# MAGIC   - [Set of ParamMaps](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder): parameters to choose from, sometimes called a _parameter grid_ to search over
# MAGIC   - [Evaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.Evaluator): metric to measure how well a fitted Model does on held-out test data
# MAGIC 
# MAGIC At a high level, model selection tools such as [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) work as follows:
# MAGIC   - They split the input data into separate training and test datasets.
# MAGIC   - For each (training, test) pair, they iterate through the set of ParamMaps:
# MAGIC     - For each [ParamMap](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder), they fit the [Estimator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.Estimator) using those parameters, get the fitted Model, and evaluate the Model's performance using the [Evaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.Evaluator).
# MAGIC   - They select the Model produced by the best-performing set of parameters.
# MAGIC 
# MAGIC The [Evaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.Evaluator) can be a [RegressionEvaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator) for regression problems. To help construct the parameter grid, users can use the [ParamGridBuilder](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder) utility.
# MAGIC 
# MAGIC Note that cross-validation over a grid of parameters is expensive. For example, in the next cell, the parameter grid has 10 values for [lr.regParam](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression.regParam), and [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) uses 3 folds. This multiplies out to (10 x 3) = 30 different models being trained. In realistic settings, it can be common to try many more parameters (e.g., multiple values for multiple parameters) and use more folds (_k_ = 3 and _k_ = 10 are common). In other words, using [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) can be very expensive. However, it is also a well-established method for choosing parameters which is more statistically sound than heuristic hand-tuning.
# MAGIC 
# MAGIC We perform the following steps:
# MAGIC   - Create a [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) using the Pipeline and [RegressionEvaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator) that we created earlier, and set the number of folds to 3
# MAGIC   - Create a list of 10 regularization parameters
# MAGIC   - Use [ParamGridBuilder](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder) to build a parameter grid with the regularization parameters and add the grid to the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)
# MAGIC   - Run the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) to find the parameters that yield the best model (i.e., lowest RMSE) and return the best model.
# MAGIC 
# MAGIC ### Exercise 7(a)
# MAGIC 
# MAGIC Run the next cell. _Note that it will take some time to run the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) as it will run almost 200 Spark jobs_

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=lrPipeline, evaluator=regEval, numFolds=3)

# Let's tune over our regularization parameter from 0.01 to 0.10
regParam = [x / 100.0 for x in range(1, 11)]

# We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, regParam)
             .build())
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
cvModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have tuned our Linear Regression model, let's see what the new RMSE and \\(r^2\\) values are versus our intial model.
# MAGIC 
# MAGIC ### Exercise 7(b)
# MAGIC 
# MAGIC Complete and run the next cell.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
# Now let's use cvModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = <FILL_IN>

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseNew = <FILL_IN>

# Now let's compute the r2 evaluation metric for our test dataset
r2New = <FILL_IN>

print("Original Root Mean Squared Error: {0:2.2f}".format(rmse))
print("New Root Mean Squared Error: {0:2.2f}".format(rmseNew))
print("Old r2: {0:2.2f}".format(r2))
print("New r2: {0:2.2f}".format(r2New))

# COMMAND ----------

# TEST
Test.assertEquals(round(rmse, 2), 4.59, "Incorrect value for rmse")
Test.assertEquals(round(rmseNew, 2), 4.59, "Incorrect value for rmseNew")
Test.assertEquals(round(r2, 2), 0.93, "Incorrect value for r2")
Test.assertEquals(round(r2New, 2), 0.93, "Incorrect value for r2New")

# COMMAND ----------

# MAGIC %md
# MAGIC So our initial untuned and tuned linear regression models are statistically identical. Let's look at the regularization parameter that the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) has selected.
# MAGIC 
# MAGIC Recall that the orginal regularization parameter we used was 0.01.
# MAGIC 
# MAGIC **NOTE**: The ML Python API currently doesn't provide a way to query the regularization parameter, so we cheat, by "reaching through" to the JVM version of the API.

# COMMAND ----------

print("Regularization parameter of the best model: {0:.2f}".format(cvModel.stages[-1]._java_obj.parent().getRegParam()))

# COMMAND ----------

# MAGIC %md
# MAGIC Given that the only linearly correlated variable is Temperature, it makes sense try another Machine Learning method such as [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning) to handle non-linear data and see if we can improve our model.
# MAGIC 
# MAGIC [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning) uses a [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree) as a predictive model which maps observations about an item to conclusions about the item's target value. It is one of the predictive modelling approaches used in statistics, data mining and machine learning. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
# MAGIC 
# MAGIC Spark ML Pipeline provides [DecisionTreeRegressor()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor) as an implementation of [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning).
# MAGIC 
# MAGIC The cell below is based on the [Spark ML Pipeline API for Decision Tree Regressor](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor).
# MAGIC 
# MAGIC ### Exercise 7(c)
# MAGIC 
# MAGIC - Read the [Decision Tree Regressor](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor) documentation
# MAGIC - In the next cell, create a [DecisionTreeRegressor()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor)
# MAGIC 
# MAGIC - The next step is to set the parameters for the method (we do this for you):
# MAGIC   - Set the name of the prediction column to "Predicted_PE"
# MAGIC   - Set the name of the features column to "features"
# MAGIC   - Set the maximum number of bins to 100
# MAGIC 
# MAGIC - Create the [ML Pipeline](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.Pipeline) and set the stages to the Vectorizer we created earlier and [DecisionTreeRegressor()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor) learner we just created.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
from pyspark.ml.regression import DecisionTreeRegressor

# Create a DecisionTreeRegressor
dt = <FILL_IN>

dt.setLabelCol("PE")\
  .setPredictionCol("Predicted_PE")\
  .setFeaturesCol("features")\
  .setMaxBins(100)

# Create a Pipeline
dtPipeline = <FILL_IN>

# Set the stages of the Pipeline
dtPipeline.<FILL_IN>

# COMMAND ----------

# TEST

Test.assertEqualsHashed(str(dtPipeline.getStages()[0].__class__.__name__), '4617be70bcf475326c0b07400b97b13457cc4949', "Incorrect pipeline stage 0")
Test.assertEqualsHashed(str(dtPipeline.getStages()[1].__class__.__name__), '46b18f257cf2f778d0d3b6e30ccc7b3398d7846a', "Incorrect pipeline stage 1")

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of guessing what parameters to use, we will use [Model Selection](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#module-pyspark.ml.tuning) or [Hyperparameter Tuning](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#module-pyspark.ml.tuning) to create the best model.
# MAGIC 
# MAGIC We can reuse the exiting [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) by replacing the Estimator with our new `dtPipeline` (the number of folds remains 3).
# MAGIC 
# MAGIC ### Exercise 7(d)
# MAGIC 
# MAGIC - Use [ParamGridBuilder](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder) to build a parameter grid with the parameter `dt.maxDepth` and a list of the values 2 and 3, and add the grid to the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)
# MAGIC - Run the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) to find the parameters that yield the best model (i.e. lowest RMSE) and return the best model.
# MAGIC 
# MAGIC _Note that it will take some time to run the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) as it will run almost 50 Spark jobs_

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
# Let's just reuse our CrossValidator with the new dtPipeline,  RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(dtPipeline)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
paramGrid = <FILL_IN>

# Add the grid to the CrossValidator
crossval.<FILL_IN>

# Now let's find and return the best model
dtModel = crossval.<FILL_IN>

# COMMAND ----------

# TEST

Test.assertEqualsHashed(str(dtModel.stages[0].__class__.__name__), '4617be70bcf475326c0b07400b97b13457cc4949', "Incorrect pipeline stage 0")
Test.assertEqualsHashed(str(dtModel.stages[1].__class__.__name__), 'a2bf7b0c1a0fb9ad35650d0478ad51a9b880befa', "Incorrect pipeline stage 1")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exercise 7(e)
# MAGIC 
# MAGIC Now let's see how our tuned DecisionTreeRegressor model's RMSE and \\(r^2\\) values compare to our tuned LinearRegression model.
# MAGIC 
# MAGIC Complete and run the next cell.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.

# Now let's use dtModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = <FILL_IN>

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseDT = <FILL_IN>

# Now let's compute the r2 evaluation metric for our test dataset
r2DT = <FILL_IN>

print("LR Root Mean Squared Error: {0:.2f}".format(rmseNew))
print("DT Root Mean Squared Error: {0:.2f}".format(rmseDT))
print("LR r2: {0:.2f}".format(r2New))
print("DT r2: {0:.2f}".format(r2DT))

# COMMAND ----------

# TEST
Test.assertEquals(round(rmseDT, 2), 5.19, "Incorrect value for rmseDT")
Test.assertEquals(round(r2DT, 2), 0.91, "Incorrect value for r2DT")

# COMMAND ----------

# MAGIC %md
# MAGIC The line below will pull the Decision Tree model from the Pipeline and display it as an if-then-else string. Again, we have to "reach through" to the JVM API to make this one work.
# MAGIC 
# MAGIC **ToDo**: Run the next cell

# COMMAND ----------

print dtModel.stages[-1]._java_obj.toDebugString()

# COMMAND ----------

# MAGIC %md
# MAGIC So our DecisionTree has slightly worse RMSE than our LinearRegression model (LR: 4.59 vs DT: 5.19). Maybe we can try an [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning) method such as [Gradient-Boosted Decision Trees](https://en.wikipedia.org/wiki/Gradient_boosting) to see if we can strengthen our model by using an ensemble of weaker trees with weighting to reduce the error in our model.
# MAGIC 
# MAGIC [Random forests](https://en.wikipedia.org/wiki/Random_forest) or random decision tree forests are an ensemble learning method for regression that operate by constructing a multitude of decision trees at training time and outputting the class that is the mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# MAGIC 
# MAGIC Spark ML Pipeline provides [RandomForestRegressor()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor) as an implementation of [Random forests](https://en.wikipedia.org/wiki/Random_forest).
# MAGIC 
# MAGIC The cell below is based on the [Spark ML Pipeline API for Random Forest Regressor](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor).
# MAGIC 
# MAGIC ### Exercise 7(f)
# MAGIC 
# MAGIC - Read the [Random Forest Regressor](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor) documentation
# MAGIC - In the next cell, create a [RandomForestRegressor()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor)
# MAGIC - The next step is to set the parameters for the method (we do this for you):
# MAGIC   - Set the name of the prediction column to "Predicted_PE"
# MAGIC   - Set the name of the features column to "features"
# MAGIC   - Set the random number generator seed to 100088121L
# MAGIC   - Set the maximum depth to 8
# MAGIC   - Set the number of trees to 30
# MAGIC - Create the [ML Pipeline](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.Pipeline) and set the stages to the Vectorizer we created earlier and [RandomForestRegressor()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor) learner we just created.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.

from pyspark.ml.regression import RandomForestRegressor

# Create a RandomForestRegressor
rf = <FILL_IN>

rf.setLabelCol("PE")\
  .setPredictionCol("Predicted_PE")\
  .setFeaturesCol("features")\
  .setSeed(100088121L)\
  .setMaxDepth(8)\
  .setNumTrees(30)

# Create a Pipeline
rfPipeline = <FILL_IN>

# Set the stages of the Pipeline
rfPipeline.<FILL_IN>

# COMMAND ----------

# TEST
Test.assertEqualsHashed(rfPipeline.getStages()[0].__class__.__name__, '4617be70bcf475326c0b07400b97b13457cc4949', "Stage 0 of pipeline is not correct")
Test.assertEqualsHashed(rfPipeline.getStages()[1].__class__.__name__, 'ecdcce2d075f00c97a6d2a2b8b1f66de322e57d2', "Stage 1 of pipeline is not correct")

# COMMAND ----------

# MAGIC %md
# MAGIC As with Decision Trees, instead guessing what parameters to use, we will use [Model Selection](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#module-pyspark.ml.tuning) or [Hyperparameter Tuning](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#module-pyspark.ml.tuning) to create the best model.
# MAGIC 
# MAGIC We can reuse the existing [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) by replacing the Estimator with our new `rfPipeline` (the number of folds remains 3).
# MAGIC 
# MAGIC ### Exercise 7(g)
# MAGIC 
# MAGIC - Use [ParamGridBuilder](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder) to build a parameter grid with the parameter `rf.maxBins` and a list of the values 50 and 100, and add the grid to the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)
# MAGIC - Run the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) to find the parameters that yield the best model (i.e., lowest RMSE) and return the best model.
# MAGIC 
# MAGIC _Note that it will take some time to run the [CrossValidator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) as it will run almost 100 Spark jobs, and each job takes longer to run than the prior CrossValidator runs._

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.
# Let's just reuse our CrossValidator with the new rfPipeline,  RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(rfPipeline)

# Let's tune over our rf.maxBins parameter on the values 50 and 100, create a parameter grid using the ParamGridBuilder
paramGrid = <FILL_IN>

# Add the grid to the CrossValidator
crossval.<FILL_IN>

# Now let's find and return the best model
rfModel = <FILL_IN>

# COMMAND ----------

# TEST
Test.assertEqualsHashed(rfModel.stages[0].__class__, 'f0c3b910468d87808e019409e7ae5e587d6aca3d', 'rfModel has incorrect stage 0')
Test.assertEqualsHashed(rfModel.stages[1].__class__, '0ed43512ea7e35ebeebeed3ddac0186248999a87', 'rfModel has incorrect stage 1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 7(h)
# MAGIC 
# MAGIC Now let's see how our tuned RandomForestRegressor model's RMSE and \\(r^2\\) values compare to our tuned LinearRegression and tuned DecisionTreeRegressor models.
# MAGIC 
# MAGIC Complete and run the next cell.

# COMMAND ----------

# TODO: Replace <FILL_IN> with the appropriate code.

# Now let's use rfModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = <FILL_IN>

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseRF = <FILL_IN>

# Now let's compute the r2 evaluation metric for our test dataset
r2RF = <FILL_IN>

print("LR Root Mean Squared Error: {0:.2f}".format(rmseNew))
print("DT Root Mean Squared Error: {0:.2f}".format(rmseDT))
print("RF Root Mean Squared Error: {0:.2f}".format(rmseRF))
print("LR r2: {0:.2f}".format(r2New))
print("DT r2: {0:.2f}".format(r2DT))
print("RF r2: {0:.2f}".format(r2RF))

# COMMAND ----------

# TEST
Test.assertEquals(round(rmseRF, 2), 3.55, "Incorrect value for rmseRF")
Test.assertEquals(round(r2RF, 2), 0.96, "Incorrect value for r2RF")

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the `r2` values are similar for all three. However, the RMSE for the Random Forest model is better.

# COMMAND ----------

# MAGIC %md
# MAGIC The line below will pull the Random Forest model from the Pipeline and display it as an if-then-else string.
# MAGIC 
# MAGIC **ToDo**: Run the next cell

# COMMAND ----------

print rfModel.stages[-1]._java_obj.toDebugString()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC 
# MAGIC Wow! So our best model is in fact our Random Forest tree model which uses an ensemble of 30 Trees with a depth of 8 to construct a better model than the single decision tree.

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
# MAGIC and change `<FILL IN>` to "CS110x-lab1":
# MAGIC 
# MAGIC ```
# MAGIC lab = "CS110x-lab1"
# MAGIC ```
# MAGIC 
# MAGIC Then, run the Autograder notebook to submit your lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ### <img src="http://spark-mooc.github.io/web-assets/images/oops.png" style="height: 200px"/> If things go wrong
# MAGIC 
# MAGIC It's possible that your notebook looks fine to you, but fails in the autograder. (This can happen when you run cells out of order, as you're working on your notebook.) If that happens, just try again, starting at the top of Appendix A.
