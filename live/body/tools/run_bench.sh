#!/bin/bash
pkill -f "python bench_compile" ; sleep 2
cd /workspace/live && nohup sh -c "python bench_compile.py none; python bench_compile.py reduce-overhead" > /workspace/logs/bench_compile2.log 2>&1 < /dev/null &
echo started
