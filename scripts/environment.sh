#!/bin/bash
# source environment.sh
CACHE_ROOT="/mnt/c/temp/__pycache__"
 
mkdir -p "$CACHE_ROOT"
 
export PYTHONPYCACHEPREFIX="$CACHE_ROOT"