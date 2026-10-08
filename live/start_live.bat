@echo off
rem Double-click: Avatar1 live (webcam -> Runpod GPU -> preview + OBS Virtual Camera)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_live.ps1" %*
