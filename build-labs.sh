#!/usr/bin/env bash
#
# Quick and dirty build script for MOOC labs.
#
# NOTE: To run this script, you'll need:
#
# 1. The master parse tool, which is in the databricks/training repo.
#    Depending on where you checked that repo out, you may need to change
#    the "master_parse" variable, below.
# 2. The "gendbc" tool, which is in the databricks/training repo, under
#    devops/gendbc. This tool is written in Scala and must be installed
#    per the README.md in its source directory. It only requires a JVM
#    to run, but this script assumes "gendbc" is in your PATH.

dir=$(dirname $0)

case "$#" in
  1)
    course=$1
    if [ ! -d src/$course ]
    then
      echo "Source directory (src/$course) doesn't exist." >&2
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 course" >&1
    echo "e.g., $0 cs105x, $0 cs120x"
    exit 1
    ;;
esac

if [ -z $MASTER_PARSE_DIR ]
then
    echo "Environment variable MASTER_PARSE_DIR is not set." >&2
    exit 1
fi
master_parse=$MASTER_PARSE_DIR/master_parse.py

rm -rf build_mp

cmd() {
  c=$1
  shift 1
  echo "$c" "$@"
  "$c" "$@"
  [ $? -eq 0 ] || exit 1
}

for i in src/$course/*.py
do
  b=$(basename $i .py)
  cmd rm -rf $course
  cmd $master_parse -ei UTF8 -eo UTF8 -db -py -in -st -cc $i
  cmd rm -f build_mp/$b/python/${b}_answers.py
  cmd mkdir $course
  cmd mv build_mp/$b/python/${b}_student.py $course/$b.py
  cmd cp $course/$b.py $b.py
  cmd rm -rf build_mp
  cmd gendbc --flatten $course $b.dbc
  cmd rm -rf $course
done
