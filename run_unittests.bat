@echo off

rem Run the install_micrograd.bat
call install_micrograd.bat

rem Activate the virtual environment
call .venv\Scripts\activate

rem Run all the unittests
python micrograd\unittests\run_unittests.py

rem Deactivate the virtual environment
deactivate