#!/usr/bin/env bash

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
