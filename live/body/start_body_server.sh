#!/usr/bin/env bash
# (Re)start the Avatar1 body server on the pod. Extra args go to pod_body_server.py.
pkill -f "python pod_body_server.py" ; sleep 2
cd /workspace/live && nohup python pod_body_server.py --port 8080 "$@" > /workspace/logs/body_server.log 2>&1 < /dev/null &
for i in $(seq 1 40); do sleep 3; grep -qE "ready on|Traceback" /workspace/logs/body_server.log && break; done
grep -E "\[body\]|Error" /workspace/logs/body_server.log | tail -5
