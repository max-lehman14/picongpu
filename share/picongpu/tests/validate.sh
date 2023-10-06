#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera, Hannes Wolf
# License: GPLv3+
#
help()
{
  echo "Validate openPMD-viewer output data."
  echo "The test is evaluating the fields, densities, particle position and -momenta."
  echo ""
  echo "Usage:"
  echo "    validate.sh [dataPath]"
  echo ""
  echo "  -d | --data dataPath                 - path to simulation output data"
  echo "                                         Default: inputPath/simOutput/simData_%T.h5"
  echo "  -h | --help                          - show help"
  echo ""
}

# options may be followed by
# - one colon to indicate they has a required argument
OPTS=`getopt -o d:h -l data:,help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# parser
while true ; do
    case "$1" in
        -d|--data)
            dataPath=$2
            shift
            ;;
        -h|--help)
            echo -e "$(help)"
            shift
            exit 0
            ;;
        --) shift; break;;
    esac
    shift
done

echo $dataPath

MAINTEST="./lib/python/test/"

if [ -z "$dataPath" ] ; then
    dataPath=$0/../simOutput/simData_%T.h5

python3 $MAINTEST/param-test.py $dataPath
exit $?
