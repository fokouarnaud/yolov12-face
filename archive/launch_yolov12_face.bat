@echo off
REM YOLOv12-Face Enhanced - Script de lancement Windows
REM =============================================

echo.
echo ===================================================
echo        YOLOv12-Face Enhanced Launcher
echo ===================================================
echo.

:menu
echo Choisissez une option:
echo.
echo 1. Preparer le dataset WIDERFace
echo 2. Entrainer le modele Enhanced (avec comparaison)
echo 3. Entrainer le modele Enhanced (sans comparaison)
echo 4. Comparer les performances (baseline vs enhanced)
echo 5. Demo webcam en temps reel
echo 6. Optimisation mobile (export)
echo 7. Entrainement rapide (50 epochs pour test)
echo 8. Workflow complet (prepare + train + compare)
echo 9. Quitter
echo.

set /p choice=Votre choix (1-9): 

if "%choice%"=="1" goto prepare_dataset
if "%choice%"=="2" goto train_compare
if "%choice%"=="3" goto train_enhanced
if "%choice%"=="4" goto compare
if "%choice%"=="5" goto webcam
if "%choice%"=="6" goto mobile
if "%choice%"=="7" goto quick_train
if "%choice%"=="8" goto full_workflow
if "%choice%"=="9" goto end

echo Choix invalide. Veuillez reessayer.
goto menu

:prepare_dataset
echo.
echo === Preparation du dataset WIDERFace ===
python scripts\prepare_widerface.py
if errorlevel 1 (
    echo.
    echo Erreur lors de la preparation du dataset!
    echo Essayez avec l'option Google Drive:
    python scripts\prepare_widerface.py --gdrive
)
pause
goto menu

:train_compare
echo.
echo === Entrainement Enhanced avec comparaison ===
set /p epochs=Nombre d'epochs (default 100): 
if "%epochs%"=="" set epochs=100
set /p batch=Batch size (default 16): 
if "%batch%"=="" set batch=16

python scripts\train_enhanced.py --compare --epochs %epochs% --batch-size %batch%
pause
goto menu

:train_enhanced
echo.
echo === Entrainement Enhanced sans comparaison ===
set /p epochs=Nombre d'epochs (default 100): 
if "%epochs%"=="" set epochs=100
set /p batch=Batch size (default 16): 
if "%batch%"=="" set batch=16

python scripts\train_enhanced.py --epochs %epochs% --batch-size %batch%
pause
goto menu

:compare
echo.
echo === Comparaison des performances ===
echo.
echo Modeles disponibles:
dir /b runs\face\*\weights\best.pt 2>nul
echo.
set /p baseline=Chemin du modele baseline: 
set /p enhanced=Chemin du modele enhanced: 

if not exist "%baseline%" (
    echo Modele baseline non trouve!
    pause
    goto menu
)
if not exist "%enhanced%" (
    echo Modele enhanced non trouve!
    pause
    goto menu
)

python scripts\compare_performance.py --baseline "%baseline%" --enhanced "%enhanced%" --save-images
pause
goto menu

:webcam
echo.
echo === Demo webcam en temps reel ===
echo.
echo Modeles disponibles:
dir /b runs\face\*\weights\best.pt 2>nul
echo.
set /p model=Chemin du modele (ou appuyez sur Entree pour le plus recent): 

if "%model%"=="" (
    for /f "tokens=*" %%i in ('dir /b /od runs\face\*enhanced*\weights\best.pt 2^>nul') do set model=%%i
)

if not exist "%model%" (
    echo Modele non trouve!
    pause
    goto menu
)

python scripts\webcam_demo.py --model "%model%" --show-fps --show-info
pause
goto menu

:mobile
echo.
echo === Optimisation mobile ===
echo.
echo Modeles disponibles:
dir /b runs\face\*\weights\best.pt 2>nul
echo.
set /p model=Chemin du modele: 

if not exist "%model%" (
    echo Modele non trouve!
    pause
    goto menu
)

echo.
echo Formats disponibles: onnx, tflite, coreml, ncnn
set /p formats=Formats a exporter (separes par des espaces): 
set /p quantize=Appliquer quantification INT8? (o/n): 

set quant_flag=
if /i "%quantize%"=="o" set quant_flag=--quantize

python scripts\mobile_optimization.py --model "%model%" --formats %formats% %quant_flag%
pause
goto menu

:quick_train
echo.
echo === Entrainement rapide (50 epochs) ===
python scripts\train_enhanced.py --compare --epochs 50 --batch-size 16
pause
goto menu

:full_workflow
echo.
echo === Workflow complet ===
echo Cette operation peut prendre plusieurs heures!
echo.
set /p confirm=Etes-vous sur de vouloir continuer? (o/n): 
if /i not "%confirm%"=="o" goto menu

echo.
echo [1/4] Preparation du dataset...
python scripts\prepare_widerface.py
if errorlevel 1 (
    echo Erreur lors de la preparation!
    pause
    goto menu
)

echo.
echo [2/4] Entrainement des modeles...
python scripts\train_enhanced.py --compare --epochs 100 --batch-size 16
if errorlevel 1 (
    echo Erreur lors de l'entrainement!
    pause
    goto menu
)

echo.
echo [3/4] Comparaison des performances...
REM Trouver les derniers modeles
for /f "tokens=*" %%i in ('dir /b /od runs\face\*baseline*\weights\best.pt 2^>nul') do set baseline=%%i
for /f "tokens=*" %%i in ('dir /b /od runs\face\*enhanced*\weights\best.pt 2^>nul') do set enhanced=%%i

python scripts\compare_performance.py --baseline "%baseline%" --enhanced "%enhanced%" --save-images

echo.
echo [4/4] Test webcam...
python scripts\webcam_demo.py --model "%enhanced%" --show-fps

echo.
echo === Workflow termine! ===
pause
goto menu

:end
echo.
echo Au revoir!
exit
