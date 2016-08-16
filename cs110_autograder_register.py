# Databricks notebook source exported at Mon, 11 Jul 2016 16:37:17 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC 
# MAGIC # Registering for the Course Autograder
# MAGIC 
# MAGIC This notebook registers you for the course autograder. You need to use the autograder to get a grade for each lab.
# MAGIC 
# MAGIC **You will only need to use this notebook once.**
# MAGIC 
# MAGIC This notebook will help you create an _autograder token_. You will use that token when you submit each lab for grading, but you'll submit each lab using the
# MAGIC [autograder notebook](https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs110_autograder.dbc).
# MAGIC 
# MAGIC If you're interested in more details on the autograder, see the [Complete Autograder notebook](https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs110_autograder_complete.dbc).
# MAGIC 
# MAGIC ## NOTE TO CS105x and CS120x STUDENTS
# MAGIC 
# MAGIC If you took CS105x or CS120x, and you already have an autograder private token, **DO NOT REGISTER FOR ANOTHER ONE.** Re-use your private token from CS105x.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Register for the course autograder
# MAGIC 
# MAGIC Enter your email address in the next cell. Your email address must be a valid email address.

# COMMAND ----------

# Replace <FILL_IN> with your email address in quotes (e.g., "tester@test.com")
username = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell. If you see an **ImportError**, you should verify that you added the `spark_mooc_meta` library to your cluster.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/autograder_LibraryError.png" alt="Drawing"  style="width: 600px;"/>

# COMMAND ----------

# Verify that the username is set
from autograder import autograder
signup = autograder()

try:
  print "Your username is " + username
except NameError:
  assert False, "Your username is not set. Please check that you set your username in the previous cell and you exectuted the cell using SHIFT-ENTER."
try:
  print "Your private token is: " + signup.signup(username)
except:
  print "autograder signup failed. please detach the cluster and re-run the notebook"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Import the Autograder Notebook
# MAGIC 
# MAGIC Import a copy of the autograder notebook:
# MAGIC 
# MAGIC 1. Download [this file](https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs110_autograder.dbc). You'll get a file called `cs110_autograder.dbc`.
# MAGIC 2. In your Databricks Community Edition account, go to your home folder, and right click on it. Select "Import", and import `cs110_autograder.dbc`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Save your private token
# MAGIC 
# MAGIC You will receive an email from the course autograder with a private token. Here is a sample email.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/autograder_signup_samplemail.png" alt="Drawing" style="width: 600px;"/>
# MAGIC 
# MAGIC Copy the private token to the clipboard. Then, go to the `cs110_autograder` notebook you uploaded in Step 2, and look for a Python cell containing:
# MAGIC 
# MAGIC ```
# MAGIC # private_token = <FILL_IN>
# MAGIC ```
# MAGIC 
# MAGIC Uncomment the cell, so you get:
# MAGIC 
# MAGIC ```
# MAGIC private_token = <FILL_IN>
# MAGIC ```
# MAGIC 
# MAGIC and replace `<FILL IN>` with the private token you just copied to the clipboard. (Be sure to surround it with quotes.)
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/autograder_private_token.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## You're ready to go.
# MAGIC 
# MAGIC You'll use the `cs110_autograder` notebook throughout the course, to submit each of your lab notebooks for grading.

# COMMAND ----------

