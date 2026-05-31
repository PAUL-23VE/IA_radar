# Sistema de Reconocimiento de Placas Vehiculares
**Materia:** Inteligencia Artificial  
**Universidad:** UTA

## Estructura del proyecto
```
proyecto_placas/
├── database/
│   ├── schema.sql          # Estructura de la BD PostgreSQL
│   ├── seed_data.sql       # Datos ficticios (50 vehículos)
│   └── db_connection.py    # Conexión y consultas a PostgreSQL
├── cnn/
│   ├── modelo.py           # Arquitectura CNN con Keras
│   ├── entrenamiento.py    # Entrenar con EMNIST Letters
│   └── inferencia.py       # Segmentar placa y predecir caracteres
├── velocidad/
│   └── logica_difusa.py    # scikit-fuzzy: velocidad → sanción
├── utils/
│   └── camara.py           # Captura de frames desde iPhone
└── main.py                 # Pipeline completo integrado
```

## Instalación
```bash
pip install tensorflow opencv-python psycopg2-binary scikit-fuzzy tensorflow-datasets numpy
```

## Cómo correr
```bash
# 1. Crear la base de datos
psql -U postgres -f database/schema.sql
psql -U postgres -d placas_db -f database/seed_data.sql

# 2. Entrenar la CNN (solo una vez, tarda ~10 min)
python cnn/entrenamiento.py

# 3. Ejecutar el sistema completo
python main.py
```
# IA_radar
