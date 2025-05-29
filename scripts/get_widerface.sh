#!/bin/bash

# Script de téléchargement et préparation du dataset WIDERFace
# Compatible avec YOLOv12 et la structure Ultralytics

echo "🚀 Préparation du dataset WIDERFace pour YOLOv12"
echo "============================================================"

# Configuration
DATASET_DIR="${1:-datasets/widerface}"
USE_GDRIVE="${2:-false}"

# Créer les répertoires
mkdir -p "$DATASET_DIR"/{images/{train,val,test},labels/{train,val,test}}

# URLs du dataset
TRAIN_URL="https://huggingface.co/datasets/wider_face/resolve/main/WIDER_train.zip"
VAL_URL="https://huggingface.co/datasets/wider_face/resolve/main/WIDER_val.zip"
ANNO_URL="http://shuoyang1213.me/WIDERFACE/support/wider_face_split.zip"

# Google Drive IDs (alternative)
TRAIN_GDRIVE="0B6eKvaijfFUOQUUwd21EckhUbWs"
VAL_GDRIVE="0B6eKvaijfFUDd3dIRmpvSk8tLUk"
ANNO_GDRIVE="1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q"

# Fonction de téléchargement
download_file() {
    local url=$1
    local output=$2
    
    if [ -f "$output" ]; then
        echo "✅ $output existe déjà"
        return 0
    fi
    
    echo "📥 Téléchargement de $output..."
    
    # Essayer wget
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$url" -O "$output" && return 0
    fi
    
    # Essayer curl
    if command -v curl &> /dev/null; then
        curl -L "$url" -o "$output" --progress-bar && return 0
    fi
    
    echo "❌ Erreur: wget ou curl requis"
    return 1
}

# Fonction de téléchargement Google Drive
download_gdrive() {
    local file_id=$1
    local output=$2
    
    if [ -f "$output" ]; then
        echo "✅ $output existe déjà"
        return 0
    fi
    
    echo "📥 Téléchargement depuis Google Drive..."
    
    # Utiliser gdown si disponible
    if command -v gdown &> /dev/null; then
        gdown "https://drive.google.com/uc?id=$file_id" -O "$output" && return 0
    fi
    
    # Alternative avec curl
    local confirm=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=$file_id" | \
                    grep -o 'confirm=[^&]*' | sed 's/confirm=//')
    
    if [ -n "$confirm" ]; then
        curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=$confirm&id=$file_id" \
             -o "$output" --progress-bar && return 0
    fi
    
    return 1
}

# Télécharger les datasets
echo "📥 Téléchargement des datasets..."

if [ "$USE_GDRIVE" = "true" ]; then
    download_gdrive "$TRAIN_GDRIVE" "$DATASET_DIR/WIDER_train.zip" || exit 1
    download_gdrive "$VAL_GDRIVE" "$DATASET_DIR/WIDER_val.zip" || exit 1
    download_gdrive "$ANNO_GDRIVE" "$DATASET_DIR/wider_face_split.zip" || exit 1
else
    download_file "$TRAIN_URL" "$DATASET_DIR/WIDER_train.zip" || \
        download_gdrive "$TRAIN_GDRIVE" "$DATASET_DIR/WIDER_train.zip" || exit 1
    
    download_file "$VAL_URL" "$DATASET_DIR/WIDER_val.zip" || \
        download_gdrive "$VAL_GDRIVE" "$DATASET_DIR/WIDER_val.zip" || exit 1
    
    download_file "$ANNO_URL" "$DATASET_DIR/wider_face_split.zip" || \
        download_gdrive "$ANNO_GDRIVE" "$DATASET_DIR/wider_face_split.zip" || exit 1
fi

# Extraire les archives
echo "📦 Extraction des archives..."
cd "$DATASET_DIR"

for zip_file in *.zip; do
    if [ -f "$zip_file" ]; then
        echo "   Extraction de $zip_file..."
        unzip -q "$zip_file" && rm "$zip_file"
    fi
done

# Vérifier que Python est disponible
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 requis pour la conversion"
    exit 1
fi

# Conversion au format YOLO
echo "🔄 Conversion au format YOLO..."
cd - > /dev/null

# Utiliser le script Python pour la conversion
python3 scripts/prepare_widerface.py --output "$DATASET_DIR" || exit 1

echo "✅ Dataset WIDERFace prêt!"
echo "📁 Emplacement: $DATASET_DIR"
echo ""
echo "💡 Pour entraîner YOLOv12:"
echo "   yolo detect train data=$DATASET_DIR/data.yaml model=yolov12n.yaml"
