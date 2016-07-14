# Databricks notebook source exported at Sun, 10 Jul 2016 19:49:03 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC # **Running Your First Notebook**
# MAGIC This notebook will show you how to install the course libraries, create your first Spark cluster, and test basic notebook functionality.  To move through the notebook just run each of the cells.  You will not need to solve any problems to complete this lab.  You can run a cell by pressing "shift-enter", which will compute the current cell and advance to the next cell, or by clicking in a cell and pressing "control-enter", which will compute the current cell and remain in that cell.
# MAGIC 
# MAGIC ** This notebook covers: **
# MAGIC * *Part 1:* Attach class helper library
# MAGIC * *Part 2:* Test Spark functionality
# MAGIC * *Part 3:* Test class helper library
# MAGIC * *Part 4:* Check plotting
# MAGIC * *Part 5:* Check MathJax formulas

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 1: Attach and test class helper library **

# COMMAND ----------

# MAGIC %md
# MAGIC #### (1a) Install class helper library into your Databricks CE workspace
# MAGIC - The class helper library "spark_mooc_meta" is published in the [PyPI Python Package repository](https://pypi.python.org/pypi) as [https://pypi.python.org/pypi/spark_mooc_meta](https://pypi.python.org/pypi/spark_mooc_meta)
# MAGIC - You can install the library into your workspace following the following instructions:
# MAGIC  - Step 1: Click on "Workspace", then on the dropdown and select "Create" and "Library"
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Library1.png" alt="Drawing" />
# MAGIC  - Step 2 Enter the name of the library by selecting "Upload Python Egg or PyPI" and entering "spark_mooc_meta" in the "PyPI Name" field
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Library2.png" alt="Drawing" />
# MAGIC  - Step 3 Make sure the checkbox for auto-attaching the library to your cluster is selected
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Library3.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 1: Test Spark functionality **

# COMMAND ----------

# MAGIC %md
# MAGIC ** (1a) Create a DataFrame and filter it **
# MAGIC 
# MAGIC When you run the next cell (with control-enter or shift-enter), you will see the following popup.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Cluster.png" alt="Drawing" />
# MAGIC 
# MAGIC Select the click box and then "Launch and Run". The display at the top of your notebook will change to "Pending"
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Cluster_Pending.png" alt="Drawing" />
# MAGIC 
# MAGIC Note that it may take a few seconds to a few minutes to start your cluster. Once your cluster is running the display will changed to "Attached"
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Cluster_Attached.png" alt="Drawing" />
# MAGIC 
# MAGIC Congratulations! You just launched your Spark cluster in the cloud!

# COMMAND ----------

# Check that Spark is working
from pyspark.sql import Row
data = [('Alice', 1), ('Bob', 2), ('Bill', 4)]
df = sqlContext.createDataFrame(data, ['name', 'age'])
fil = df.filter(df.age > 3).collect()
print fil

# If the Spark job doesn't work properly this will raise an AssertionError
assert fil == [Row(u'Bill', 4)]

# COMMAND ----------

# MAGIC %md
# MAGIC ** (2b) Loading a text file **
# MAGIC 
# MAGIC Let's load a text file.

# COMMAND ----------

# Check loading data with sqlContext.read.text
import os.path
baseDir = os.path.join('databricks-datasets', 'cs100')
inputPath = os.path.join('lab1', 'data-001', 'shakespeare.txt')
fileName = os.path.join(baseDir, inputPath)

dataDF = sqlContext.read.text(fileName)
shakespeareCount = dataDF.count()

print shakespeareCount

# If the text file didn't load properly an AssertionError will be raised
assert shakespeareCount == 122395

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 3: Test class testing library **

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3a) Compare with hash **
# MAGIC 
# MAGIC Run the following cell. If you see an **ImportError**, you should verify that you added the spark_mooc_meta library to your cluster and, if necessary, repeat step (1a).
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_LibraryError.png" alt="Drawing"  style="width: 600px;"/>

# COMMAND ----------

# TEST Compare with hash (2a)
# Check our testing library/package
# This should print '1 test passed.' on two lines
from databricks_test_helper import Test

twelve = 12
Test.assertEquals(twelve, 12, 'twelve should equal 12')
Test.assertEqualsHashed(twelve, '7b52009b64fd0a2a49e6d8a939753077792b0554',
                        'twelve, once hashed, should equal the hashed value of 12')

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3b) Compare lists **

# COMMAND ----------

# TEST Compare lists (2b)
# This should print '1 test passed.'
unsortedList = [(5, 'b'), (5, 'a'), (4, 'c'), (3, 'a')]
Test.assertEquals(sorted(unsortedList), [(3, 'a'), (4, 'c'), (5, 'a'), (5, 'b')],
                  'unsortedList does not sort properly')

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
# MAGIC and change `<FILL IN>` to "CS120x-lab0":
# MAGIC 
# MAGIC ```
# MAGIC lab = "CS120x-lab0"
# MAGIC ```
# MAGIC 
# MAGIC Then, run the Autograder notebook to submit your lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ### <img src="http://spark-mooc.github.io/web-assets/images/oops.png" style="height: 200px"/> If things go wrong
# MAGIC 
# MAGIC It's possible that your notebook looks fine to you, but fails in the autograder. (This can happen when you run cells out of order, as you're working on your notebook.) If that happens, just try again, starting at the top of Appendix A.
