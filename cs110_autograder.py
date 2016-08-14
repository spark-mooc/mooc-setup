# Databricks notebook source exported at Mon, 11 Jul 2016 17:51:44 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC 
# MAGIC # The Course Autograder
# MAGIC 
# MAGIC Grading in this course is a three step process:
# MAGIC 
# MAGIC 1. **Grade Locally.** Completing the lab exercises and using the lab notebook's built-in autograder functionality (TestAssert). The built-in autograder provides you with immediate feedback about the correctness of your code. You will spend most of your time using this built-in autograder functionality.
# MAGIC 2. **Submit to Autograder.** Using the autograder notebook to submit a completed lab exercise to the course autograder. The course autograder runs the exact same tests as the built-in notebook autograder. There is no advantage to submitting to the course autograder until your code passes as many built-in tests as you can.
# MAGIC 3. **Submit to edX.** Submitting the submission ID from the course autograder on the edX course website. Submitting your submission ID on the edX course website will record your score for the assignment. You should only need to do this step once, after you have completed a lab exercise.
# MAGIC 
# MAGIC ** You must complete _all_ three steps to receive a score for a lab assignment. **
# MAGIC 
# MAGIC This notebook will guide you through Steps 2 and 3 of this process.
# MAGIC 
# MAGIC **You _must_ register with the autograder first.**
# MAGIC 
# MAGIC If you have not yet registered with the autograder, download a copy of [the autograder registration notebook](https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs110_autograder_register.dbc) and follow the instructions.
# MAGIC 
# MAGIC ## A note on running this notebook
# MAGIC 
# MAGIC You can run a cell by pressing "shift-enter", which will compute the current cell and advance to the next cell, or by clicking in a cell and pressing "control-enter", which will compute the current cell and remain in that cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section A: Put your registration token in the following cell
# MAGIC 
# MAGIC This notebook _will not work_ unless the following cell is filled properly. See
# MAGIC [the autograder registration notebook](https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs110_autograder_register.dbc)
# MAGIC for details.

# COMMAND ----------

# AFTER you receive an email from the autograder:
#    1. Replace <FILL_IN> with your private token in quotes (e.g., "ABCDEFGHIJKLM")
#    2. Remove the comment symbol from before private_token
#    3. Press SHIFT-ENTER to run this cell

# private_token = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section B: Performing Step 2: Submit to Autograder
# MAGIC 
# MAGIC ### Step B-1: Publish your notebook
# MAGIC 
# MAGIC At the bottom of your lab notebook, you'll find instructions on how to publish your notebook. Follow those instructions _carefully_.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B-2: Paste the notebook's published URL in the cell below

# COMMAND ----------

# Set the published notebook_url (e.g., notebook_url = "https://databricks-prod-cloudfront.cloud.databricks.com/public/....")
notebook_url = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B-3: Add your user name
# MAGIC 
# MAGIC Put your user name (the email address you used in the [the autograder registration notebook](https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs110_autograder_register.dbc)) in the cell below.

# COMMAND ----------

username = <FILL IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B-4: Set the lab ID
# MAGIC 
# MAGIC Get the Lab ID from your lab notebook. It's in the section at the bottom of the notebook, in the Autograder Appendix, in a cell entitled "Step 2: Set the notebook URL and Lab ID in the Autograder notebook, and run it".
# MAGIC Set the lab ID in the cell below.
# MAGIC 
# MAGIC The lab ID is used to select the appropriate autograder queue. If you get the lab ID wrong, the autograder will complain.

# COMMAND ----------

# Set lab variable (e.g., lab = "CS110x-lab1")
lab = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B-5: Re-run the AUTOGRADER notebook (_this_ notebook)
# MAGIC 
# MAGIC Click on "Run All" at the top of this notebook.
# MAGIC 
# MAGIC  <img src="http://spark-mooc.github.io/web-assets/images/submit_runall.png" alt="Drawing" />
# MAGIC 
# MAGIC This step _should_ finish quickly.
# MAGIC 
# MAGIC Wait for your cluster to finish running the cells in your autograder notebook before proceeding.

# COMMAND ----------

# Verify that the username is set
try:
  print "Your username is " + username
except NameError:
  assert False, "Your username is not set. Please ensure that you set your username in the cell at the beginning of the notebook and you exectuted the cell using SHIFT-ENTER."

# COMMAND ----------

# Verify that the private_token is set
try:
  print "Your private token is " + private_token
except NameError:
  assert False, "Your private token is not set. Please ensure that you set the private_token in the cell at the beginning of the notebook, removed the comment character (#) before private_token, and you exectuted the cell using SHIFT-ENTER."

# COMMAND ----------

# Verify that the lab ID is set
try:
  print "You are submitting " + lab
except NameError:
  assert False, "Your lab ID value is not set. Please check that you set the lab variable, above."

# COMMAND ----------

from autograder import autograder

client = autograder(username, private_token)
result = client.submit(lab, notebook_url)
print "Result for autograder#submit(): %s" % result

# COMMAND ----------

# MAGIC %md
# MAGIC Execution will stop at the next cell, which has a _deliberate_ error.
# MAGIC 
# MAGIC **If all the cells above worked, then your lab has been submitted to the autograder.**
# MAGIC 
# MAGIC To submit your lab to edX, continue below.

# COMMAND ----------

stop_here # This will fail. That's okay.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section C: Performing Step 3: Submit to edX
# MAGIC 
# MAGIC To get a grade for your lab on edX, you have to complete the following steps. You'll do this for every lab.
# MAGIC 
# MAGIC ### Step C-1: Get your submission ID for submitting on the edX course website
# MAGIC 
# MAGIC To record your score for a lab, you need to submit a submission ID on the edX course website.
# MAGIC 
# MAGIC 1. Run the following cell to get a list of submission IDs for each lab you have submitted to the autograder.
# MAGIC 2. Copy the `submission_id` for the lab you want to submit to edX for a score.
# MAGIC 3. On the edX website page for the lab you completing, enter your **username for autograder** and **submission ID for the lab** to receive a score for the assignment on edX.

# COMMAND ----------

import json
(result,submission_list) = client.get_submission_list(lab)
print "Result for get_submission_list(): %s" % result
if (submission_list == []):
  print "No submissions found for lab of %s. Please re-run the notebook and check the output in B-5." % lab
else:
  # convert result to a Spark DataFrame
  df_submission_list = sqlContext.jsonRDD(sc.parallelize([json.dumps(item) for item in submission_list]))
  print "Pick up one submission ID with your expected grade and submit it to edX for your final grade."
  display(df_submission_list['submission_timestamp','grade','submission_id','lab','autograder_results','username'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section D: Checking the status of the course autograder queue (OPTIONAL)
# MAGIC 
# MAGIC To check the status of the autograder queue, just run the following cell. You can run it more than once.
# MAGIC 
# MAGIC When the queue is empty for you, it means all your submissions have been graded by the autograder, and you can move on to submitting to edX.
# MAGIC 
# MAGIC **Note:** It will typically take a few minutes before you receive autograder feedback. If you do not receive feedback within one hour, please use the [Piazza discussion group](http://piazza.com/edx_berkeley/fall2016/cs110x/home) to contact the TAs for support.

# COMMAND ----------

# Re-run this cell to see the autograder queue status
import json
(result,queue) = client.get_queue_status()
print "Result for get_queue_status(): %s" % result
if (queue == []):
  print "All submissions are processed. Proceed to Part 5."
else:
  # convert result to a Spark DataFrame
  print "If there are no submissions in the queue with your name.Proceed to Part 5."
  df_queue = sqlContext.jsonRDD(sc.parallelize([json.dumps(item) for item in queue]))
  display(df_queue['submission_timestamp','grading_status','lab','username'])
