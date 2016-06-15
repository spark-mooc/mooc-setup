# Databricks notebook source exported at Wed, 15 Jun 2016 17:01:17 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC 
# MAGIC # **Using the Course Autograder**
# MAGIC 
# MAGIC Grading in this course is a three step process:
# MAGIC * Completing the lab exercises and using the lab notebook's built-in autograder functionality (TestAssert). The built-in autograder provides you with immediate feedback about the correctness of your code.
# MAGIC * Using the autograder notebook to submit a completed lab exercise to the course autograder. The course autograder runs the _exact_ same tests as the built-in notebook autograder. _There is no advantage to submitting to the course autograder until your code passes all built-in tests._
# MAGIC * Submitting the submission ID from the course autograder on the edX course website. Submitting your submission ID on the edX course website will record your score for the assignment.
# MAGIC 
# MAGIC ** You must complete _all_ three steps to receive a score for a lab assignment. **
# MAGIC 
# MAGIC This notebook will show you how to register for the course autograder, prepare a lab notebook for submission to the course autograder, and submit a notebook for grading by the autograder.
# MAGIC 
# MAGIC You can run a cell by pressing "shift-enter", which will compute the current cell and advance to the next cell, or by clicking in a cell and pressing "control-enter", which will compute the current cell and remain in that cell.
# MAGIC 
# MAGIC #### ** Make sure you have completed the Lab 0 notebook before you run this notebook  - this notebook uses libraries that are installed when you complete Lab 0. **
# MAGIC 
# MAGIC ** This notebook covers: **
# MAGIC * *Part 1:* Register for the course autograder
# MAGIC * *Part 2:* Save your private token
# MAGIC * *Part 3:* Submit a lab to the course autograder
# MAGIC * *Part 4:* Check the status of the course autograder queue
# MAGIC * *Part 5:* Get your submission ID for submitting on the edX course website
# MAGIC * *Part 6:* Examine the autograder results for your submission

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ignore the following cell. You will fill it in after Part 2 after you have received the email from the autograder

# COMMAND ----------

# AFTER you receive an email from the autograder:
#    1. Replace <FILL_IN> with your private token in quotes (e.g., "ABCDEFGHIJKLM")
#    2. Remove the comment symbol from before private_token
#    3. Press SHIFT-ENTER to run this cell

# private_token = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 1: Register for the course autograder **
# MAGIC 
# MAGIC Enter your email address in the next cell. Your email address must be a valid email address.

# COMMAND ----------

# Replace <FILL_IN> with your email address in quotes (e.g., "tester@test.com")
username = <FILL_IN>

# COMMAND ----------

# Verify that the username is set
try:
  print "Your username is " + username
except NameError:
  assert False, "Your username is not set. Please check that you set your username in the previous cell and you exectuted the cell using SHIFT-ENTER."

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell. If you see an **ImportError**, you should verify that you added the spark_mooc_meta library to your cluster and, if necessary, repeat step (1a) in **cs105_lab0.**
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/autograder_LibraryError.png" alt="Drawing"  style="width: 600px;"/>

# COMMAND ----------

from autograder import autograder
signup = autograder()
try:
  print "Your private token is " + private_token
except NameError:
  print signup.signup(username)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 2: Save your private token **
# MAGIC 
# MAGIC You will receive an email from the course autograder with a private token. Here is a sample email.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/autograder_signup_samplemail.png" alt="Drawing" style="width: 600px;"/>
# MAGIC 
# MAGIC Copy the private token and paste it into the cell at the beginning of the notebook. Make sure you remove the comment character (#) before *private_token* and you execute the cell using SHIFT-ENTER.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/autograder_private_token.png" alt="Drawing" />

# COMMAND ----------

# Verify that the private_token is set
try:
  print "Your private token is " + private_token
except NameError:
  assert False, "Your private token is not set. Please check that you: set the private_token in the cell at the beginning of the notebook, removed the comment character (#) before private_token, and you exectuted the cell using SHIFT-ENTER."

# COMMAND ----------

# MAGIC %md
# MAGIC Now, you can use this notebook to submit a lab to the course autograder, get a list of your lab submissions to the autograder, get details about a specific submission you made, and get the status of the autograder queue.

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 3: Submit a lab to the course autograder **
# MAGIC 
# MAGIC Once you confirm that your lab notebook is passing all tests, you can submit it first to the course autograder and then second to the edX website to receive a grade.
# MAGIC 
# MAGIC To perform the steps in Part 3, open two browser tabs: one with your lab notebook, and one with this autograder notebook.
# MAGIC 
# MAGIC ** Note that you can only submit to the course autograder once every ten minutes. **
# MAGIC 
# MAGIC ### Every time you submit to the course autograder, you must perform steps (3a), (3b), (3c), and (3d). ##

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3a) Restart your cluster by clicking on the dropdown next to your cluster name and selecting "Restart Cluster".**
# MAGIC 
# MAGIC You can do this step in either notebook, since there is one cluster for your notebooks.
# MAGIC 
# MAGIC  <img src="http://spark-mooc.github.io/web-assets/images/submit_restart.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3b) _IN YOUR LAB NOTEBOOK_, click on "Run All" to run all of the cells in your LAB notebook. **
# MAGIC 
# MAGIC   <img src="http://spark-mooc.github.io/web-assets/images/submit_runall.png" alt="Drawing" />
# MAGIC 
# MAGIC This step will take some time. While the cluster is running all the cells in your lab notebook, you will see the "Stop Execution" button.
# MAGIC 
# MAGIC   <img src="http://spark-mooc.github.io/web-assets/images/stop_execution.png" alt="Drawing" />
# MAGIC 
# MAGIC Wait for your cluster to finish running the cells in your lab notebook before proceeding.

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3c) Verify that your LAB notebook passes as many tests as you can. **
# MAGIC 
# MAGIC Most computations should complete within a few seconds unless stated otherwise. As soon as the expression of a cell have been successfully evaluated, you will see one or more "test passed" messages if the cell includes test expressions:
# MAGIC 
# MAGIC   <img src="http://spark-mooc.github.io/web-assets/images/test_passed.png" alt="Drawing" />
# MAGIC 
# MAGIC or just execution time otherwise:
# MAGIC 
# MAGIC   <img src="http://spark-mooc.github.io/web-assets/images/execution_time.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3d) Publish your LAB notebook by clicking on the "Publish" button at the top of your LAB notebook. **
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish0.png" alt="Drawing" />
# MAGIC 
# MAGIC When you click on the button, you will see the following popup.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish1.png" alt="Drawing" />
# MAGIC 
# MAGIC When you click on "Publish", you will see a popup with your notebook's public link. __Copy the link and set the notebook_URL variable in the next cell to the link.__
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish2.png" alt="Drawing" />

# COMMAND ----------

# Set the published notebook_url (e.g., notebook_url = "https://databricks-prod-cloudfront.cloud.databricks.com/public/....")
notebook_url = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3e) Re-run the AUTOGRADER notebook **
# MAGIC 
# MAGIC Because you just restarted your cluster, all of the state of this autograder notebook has been lost. To recover it, all you have to do is to click on "Run All" (just as you did for your Lab notebook) and your cluster will run all the cells.
# MAGIC 
# MAGIC  <img src="http://spark-mooc.github.io/web-assets/images/submit_runall.png" alt="Drawing" />
# MAGIC 
# MAGIC This step will finish quickly. While the cluster is running all the cells in your autograder notebook, you will see the "Stop Execution" button.
# MAGIC 
# MAGIC   <img src="http://spark-mooc.github.io/web-assets/images/stop_execution.png" alt="Drawing" />
# MAGIC 
# MAGIC Wait for your cluster to finish running the cells in your autograder notebook before proceeding.
# MAGIC 
# MAGIC Note that it will stop on Step (3f), since you have not set the _lab_ variable yet.

# COMMAND ----------

# Verify that the username is set
try:
  print "Your username is " + username
except NameError:
  assert False, "Your username is not set. Please check that you set your username in the cell at the beginning of the notebook and you exectuted the cell using SHIFT-ENTER."

# COMMAND ----------

# Verify that the private_token is set
try:
  print "Your private token is " + private_token
except NameError:
  assert False, "Your private token is not set. Please check that you: set the private_token in the cell at the beginning of the notebook, removed the comment character (#) before private_token, and you exectuted the cell using SHIFT-ENTER."

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3f) Select the Course Autograder queue **
# MAGIC 
# MAGIC There are three autograder queues, one for each lab assignment in CS105x: **CS105x-lab0** (setup), **CS105x-lab1b**, and **CS105x-lab2**.
# MAGIC Set the *lab* variable to the assignment you are submitting.

# COMMAND ----------

# Set lab variable (e.g., lab = "CS105x-lab0")
lab = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ** (3g) Submit your published notebook to the Course Autograder queue **

# COMMAND ----------

client = autograder(username, private_token)
client.submit(lab, notebook_url)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 4: Check the status of the course autograder queue **
# MAGIC 
# MAGIC To check the status of the autograder queue, you use **get_queue_status()**.
# MAGIC 
# MAGIC You can re-run the following cell to redisplay the autograder queue status.
# MAGIC 
# MAGIC When the queue is empty for you, it means all your submissions have been graded by the autograder and you can proceed to Part 5.
# MAGIC 
# MAGIC **Note:** It will typically take a few minutes before you receive autograder feedback. If you do not receive feedback within one hour, please use the [Piazza discussion group](https://piazza.com/edx_berkeley/summer2016/cs1051x) to contact the TAs for support.

# COMMAND ----------

# Re-run this cell to see the autograder queue status
import json
(result,queue) = client.get_queue_status()
if (queue == []):
  print "No submisions for %s found in autograder queue. Proceed to Part 5." % username
else:
  # convert result to a Spark DataFrame
  df_queue = sqlContext.jsonRDD(sc.parallelize([json.dumps(item) for item in queue]))
  display(df_queue['submission_timestamp','grading_status','lab','username'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 5: Get your submission ID for submitting on the edX course website **
# MAGIC 
# MAGIC To record your score for a lab, you need to submit a submission ID on the edX course website. Use **get_submission_list(lab)** to check your autograder results.
# MAGIC 
# MAGIC * Copy the _submission_id_ that you want to submit for a score.
# MAGIC * On the edX website page for the lab you completing, enter your **username for autograder** and **submission ID for the lab** to receive a score for the assignment on edX.

# COMMAND ----------

import json
(result,submission_list) = client.get_submission_list(lab)
if (submission_list == []):
  print "No submisions for %s found in autograder queue for lab %s" % (username, lab)
else:
  # convert result to a Spark DataFrame
  df_submission_list = sqlContext.jsonRDD(sc.parallelize([json.dumps(item) for item in submission_list]))
  display(df_submission_list['submission_timestamp','grade','submission_id','lab','username'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### ** Part 6: Examine the autograder results for your submission **
# MAGIC 
# MAGIC You can examine the feedback from the autograder using a _submission_id_ and **get_submission_detail(submission_id)**.
# MAGIC 
# MAGIC * Copy the _submission_id_ that you want to submit for a score.
# MAGIC * On the edX website page for the lab you completing, enter your **username for autograder** and **submission ID for the lab** to receive a score for the assignment on edX.

# COMMAND ----------

# Set the submission_id (e.g., submission_id = "CS105x-lab0-...-...-...-...-....")
submission_id = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ** (6a) Viewing the number of tests passed and score **

# COMMAND ----------

import json
(result,submission_detail) = client.get_submission_detail(submission_id)
print submission_detail['autograder_results']
print submission_detail['grade']

# COMMAND ----------

# MAGIC %md
# MAGIC ** (6b) Viewing the raw output from the autograder **
# MAGIC 
# MAGIC You can examine the complete output from the autograder.

# COMMAND ----------

# Use displayHTML() function to check the raw results returned from autograder.
displayHTML(html = submission_detail['raw_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## You will use this notebook throughout the course to submit your lab exercises.
# MAGIC 
# MAGIC All you have to do is follow the steps in Part 3 and then click "Run All" to run all the cells in this notebook

# COMMAND ----------


