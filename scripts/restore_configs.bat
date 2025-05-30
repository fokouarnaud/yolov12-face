@echo off
REM ========================================================
REM 🔄 Script de Restauration YOLOv12-Face Enhanced (Windows)
REM ========================================================
REM
REM Ce script restaure automatiquement les fichiers de
REM configuration après une réinstallation d'Ultralytics.
REM
REM Usage: double-clic ou 'restore_configs.bat'
REM ========================================================

echo.
echo 🚀 YOLOv12-Face Enhanced - Restauration des Configurations
echo ============================================================

REM Vérifier si nous sommes dans le bon répertoire
if not exist "scripts\configs" (
    echo ❌ Erreur: Dossier 'scripts\configs' non trouvé
    echo 💡 Exécutez ce script depuis le répertoire racine du projet
    pause
    exit /b 1
)

echo 📁 Répertoire de travail: %CD%
echo.

REM Créer les dossiers de destination si nécessaire
echo 🔧 Création des dossiers de destination...
if not exist "ultralytics\cfg\datasets" mkdir "ultralytics\cfg\datasets"
if not exist "ultralytics\cfg\models\v12" mkdir "ultralytics\cfg\models\v12"
if not exist "ultralytics\nn\modules" mkdir "ultralytics\nn\modules"

REM Copier les fichiers de configuration
echo.
echo 🔄 Restauration des fichiers...

if exist "scripts\configs\datasets\widerface.yaml" (
    copy "scripts\configs\datasets\widerface.yaml" "ultralytics\cfg\datasets\widerface.yaml" >nul
    echo ✅ Restauré: ultralytics\cfg\datasets\widerface.yaml
) else (
    echo ⚠️  Manquant: scripts\configs\datasets\widerface.yaml
)

if exist "scripts\configs\models\v12\yolov12-face.yaml" (
    copy "scripts\configs\models\v12\yolov12-face.yaml" "ultralytics\cfg\models\v12\yolov12-face.yaml" >nul
    echo ✅ Restauré: ultralytics\cfg\models\v12\yolov12-face.yaml
) else (
    echo ⚠️  Manquant: scripts\configs\models\v12\yolov12-face.yaml
)

if exist "scripts\configs\models\v12\yolov12-face-enhanced.yaml" (
    copy "scripts\configs\models\v12\yolov12-face-enhanced.yaml" "ultralytics\cfg\models\v12\yolov12-face-enhanced.yaml" >nul
    echo ✅ Restauré: ultralytics\cfg\models\v12\yolov12-face-enhanced.yaml
) else (
    echo ⚠️  Manquant: scripts\configs\models\v12\yolov12-face-enhanced.yaml
)

if exist "scripts\configs\modules\enhanced.py" (
    copy "scripts\configs\modules\enhanced.py" "ultralytics\nn\modules\enhanced.py" >nul
    echo ✅ Restauré: ultralytics\nn\modules\enhanced.py
) else (
    echo ⚠️  Manquant: scripts\configs\modules\enhanced.py
)

echo.
echo 📋 Mise à jour de __init__.py...

REM Note: La mise à jour de __init__.py nécessite Python
REM L'utilisateur devra le faire manuellement ou utiliser le script Python

if exist "ultralytics\nn\modules\__init__.py" (
    echo ℹ️  Fichier __init__.py trouvé
    echo 💡 Ajoutez manuellement cette ligne avant __all__:
    echo     from .enhanced import *
) else (
    echo ⚠️  Fichier __init__.py non trouvé
    echo 💡 Installez d'abord Ultralytics: pip install ultralytics
)

echo.
echo ✅ Restauration terminée !
echo.
echo 🧪 Pour tester la configuration:
echo    python -c "from ultralytics.nn.modules.enhanced import A2Module; print('✅ OK')"
echo.
echo 📖 Pour plus d'informations:
echo    Consultez scripts\configs\README_RESTORATION.md
echo.

pause
