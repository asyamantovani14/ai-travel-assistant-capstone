@echo off
REM Spostati nella cartella src
cd /d "C:\Users\asyam\OneDrive\Documenti\GitHub\ai-travel-assistant-capstone\src"

REM Esegui lo script con Python del venv e salva output e errori in logs\pipeline.log
"C:\Users\asyam\OneDrive\Documenti\GitHub\ai-travel-assistant-capstone\venv\Scripts\python.exe" run_daily_pipeline.py >> "..\logs\pipeline.log" 2>&1
