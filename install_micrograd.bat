@echo off

if "%~1" equ "--skip_install" (
    pip show micrograd >nul 2>&1
    if %errorlevel% equ 0 (
        rem Echo that we are skipping the installation
        echo Skipping the installation
        goto :skip
    )
)

rem Check if the .venv directory exists
if not exist .venv (
    rem Create a virtual environment
    python -m venv .venv
)

rem Activate the virtual environment
call .venv\Scripts\activate

rem Echo that we are installing the package
echo Installing the package

rem Check if the .hash file exists
if not exist .hash (
    rem Write an empty string to the .hash file
    echo.>.hash
)

rem Read the last source hash from the file
set /p last_hash=<.hash

rem Create a hash of the source files using the hash_source.py script
python hash_source.py

rem Read the hash from the file
set /p hash=<.hash

rem Check if the first command line argument is "reinstall"
if "%~1" equ "--reinstall" (
    rem Echo that we are skipping the installation
    echo Performing forced installation
    goto :install
)

rem Check if the package is not already installed
pip show micrograd >nul 2>&1
if %errorlevel% neq 0 (
    rem Echo that we are skipping the installation
    echo Performing forced installation
    goto :install
)

rem If the hash is not empty then perform the check for skip installation
if "%hash%" neq "" (
    rem check if the last source hash is the same as the current source hash
    if "%last_hash%" equ "%hash%" (
        rem Echo that we are skipping the installation
        echo Skipping the installation
        rem Write back the last source hash to the file
        echo %last_hash%>.hash
        goto :skip
    )
)

:install

rem Install the build module
python -m pip install build

rem remove the old wheel files
if exist dist\*.whl (
    del /Q dist\*.whl
)

rem remove the old egg-info files
if exist micrograd.egg-info (
    del /Q micrograd.egg-info\*
)

rem Uninstall the old package if any
pip show micrograd >nul 2>&1
if %errorlevel% equ 0 (
    python -m pip uninstall -y micrograd
)

rem Build the package
python -m build

rem Install the required packages
python -m pip install -r micrograd.egg-info\requires.txt

rem Install the package using installer
python -m pip install installer
for %%f in (dist\*.whl) do (
    python -m installer %%f
)

rem Check if the PYTHONPATH contains the micrograd directory
echo %PYTHONPATH% | findstr /C:"%CD%\micrograd" >nul 2>&1

rem If the PYTHONPATH does not contain the micrograd directory then add the path
if %errorlevel% neq 0 (
    rem Add the path
    rem Check if the PYTHONPATH is empty and do not add the semicolon if it is empty
    if "%PYTHONPATH%" equ "" (
        set "PYTHONPATH=%CD%\micrograd"
    ) else (
        set "PYTHONPATH=%PYTHONPATH%;%CD%\micrograd"
    )
)

rem Echo the current pythonpath
echo PYTHONPATH=%PYTHONPATH%

rem Check if the PATH contains the current directory
echo %PATH% | findstr /C:"%CD%\micrograd" >nul 2>&1

rem If the path does not contain the current directory then add the path
if %errorlevel% neq 0 (
    rem Add the path
    rem Check if the PATH is empty and do not add the semicolon if it is empty
    if "%PATH%" equ "" (
        set "PATH=%CD%\micrograd"
    ) else (
        set "PATH=%PATH%;%CD%\micrograd"
    )
)

rem Echo the current path
echo PATH=%PATH%

rem Deactivate the virtual environment
deactivate

:skip