@echo off
REM Script de téléchargement et préparation du dataset WIDERFace pour Windows
REM Compatible avec YOLOv12 et la structure Ultralytics

echo ======================================================
echo 🚀 Preparation du dataset WIDERFace pour YOLOv12
echo ======================================================

REM Configuration
set DATASET_DIR=%1
if "%DATASET_DIR%"=="" set DATASET_DIR=datasets\widerface

REM Créer les répertoires
echo 📁 Creation des repertoires...
mkdir "%DATASET_DIR%\images\train" 2>nul
mkdir "%DATASET_DIR%\images\val" 2>nul
mkdir "%DATASET_DIR%\images\test" 2>nul
mkdir "%DATASET_DIR%\labels\train" 2>nul
mkdir "%DATASET_DIR%\labels\val" 2>nul
mkdir "%DATASET_DIR%\labels\test" 2>nul

REM Vérifier Python
echo 🐍 Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installe ou n'est pas dans le PATH
    echo 💡 Veuillez installer Python depuis https://www.python.org/
    pause
    exit /b 1
)

REM Installer les dépendances si nécessaire
echo 📦 Installation des dependances...
pip install requests tqdm opencv-python pyyaml gdown >nul 2>&1

REM Exécuter le script Python
echo 🔄 Execution du script de preparation...
python scripts\prepare_widerface.py --output "%DATASET_DIR%"

if errorlevel 1 (
    echo ❌ Erreur lors de la preparation du dataset
    pause
    exit /b 1
)

echo.
echo ✅ Dataset WIDERFace pret!
echo 📁 Emplacement: %DATASET_DIR%
echo.
echo 💡 Pour entrainer YOLOv12:
echo    yolo detect train data=%DATASET_DIR%\data.yaml model=yolov12n.yaml
echo.
pause
