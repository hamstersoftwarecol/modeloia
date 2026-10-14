#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=================================================="
echo " Configuración del Entorno para Mini-Transformer "
echo "=================================================="

# 1. Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "[*] Creando entorno virtual Python en $DIR/venv..."
    python3 -m venv venv
else
    echo "[*] El entorno virtual 'venv' ya existe."
fi

# 2. Activar entorno virtual
echo "[*] Activando entorno virtual..."
source venv/bin/activate

# 3. Actualizar pip e instalar requerimientos
echo "[*] Instalando dependencias (PyTorch)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================================="
echo " ¡Instalación completada con éxito! "
echo "=================================================="
echo "Para activar el entorno y usar el modelo, ejecuta:"
echo "  cd $DIR"
echo "  source venv/bin/activate"
echo ""
echo "Para entrenar el modelo:"
echo "  python train.py"
echo ""
echo "Para generar texto:"
echo "  python generate.py --prompt \"En un lugar de la Mancha\""
echo "=================================================="
