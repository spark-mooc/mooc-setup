#!/usr/bin/env bash
#
# Quick and dirty build script for CS105.X
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

master_parse=$dir/../training/devops/master_parse/master_parse.py

rm -rf build_mp

cmd() {
  c=$1
  shift 1
  echo "$c" "$@"
  "$c" "$@"
  [ $? -eq 0 ] || exit 1
}

for i in src/*.py
do
  b=$(basename $i .py)
  cmd rm -rf cs105
  cmd $master_parse -db -py -in -st -cc $i
  cmd rm -f build_mp/$b/python/${b}_answers.py
  cmd mkdir cs105
  cmd mv build_mp/$b/python/${b}_student.py cs105/$b.py
  cmd cp cs105/$b.py $b.py
  cmd rm -rf build_mp
  cmd gendbc --flatten cs105 $b.dbc
  cmd rm -rf cs105
done
