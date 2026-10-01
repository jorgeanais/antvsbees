# antvsbees

>  Clasificador de Hormigas vs Abejas (ConvNeXt Base + Transfer Learning) App Streamlit para inferencia sobre una imagen subida por el usuario.

![example](/home/jorge/Documents/code/antvsbees/example.png)

## Estructura

```
clasificador-hormigas-abejas/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── celda_exportar_modelo.py   # celda para el notebook
└── model/
    └── modelo_hormigas_abejas.pth   # <- lo generas desde el notebook
```

## 1. Exportar el modelo desde el notebook
Ejecuta la celda de `celda_exportar_modelo.py` al final del notebook
(después del fine-tuning) y copia el `.pth` dentro de `model/`.

## 2. Construir y ejecutar
```bash
docker build -t clasificador-hormigas-abejas .
docker run --rm -p 8501:8501 clasificador-hormigas-abejas
```
o con compose:
```bash
docker compose up --build
```
Abre http://localhost:8501

## Probar sin Docker
```bash
pip install torch torchvision -r requirements.txt
streamlit run app.py
```
