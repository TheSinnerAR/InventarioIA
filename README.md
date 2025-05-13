# Proyecto Inventario con IA

Este repositorio contiene una prueba de concepto para la gestión predictiva de inventarios usando IA:

## Descripción del Proyecto
La Inteligencia Artificial (IA) es una disciplina de la informática dedicada a desarrollar sistemas capaces de ejecutar tareas que tradicionalmente requieren intervención humana, tales como la automatización de procesos, la optimización de recursos y el soporte a la toma de decisiones estratégicas basadas en datos.
### Objetivo
El objetivo de este proyecto es diseñar y planificar una solución basada en IA que optimice significativamente el proceso de gestión de inventarios de una empresa. La herramienta propuesta:
- Maximiza la eficiencia mediante la automatización inteligente de la reposición de productos.
- Reduce los costos asociados al inventario.
- Monitorea continuamente los niveles de stock para prevenir productos obsoletos.

## Implementación del Pipeline (`src/implementacion_pipeline.py`)
El archivo `src/implementacion_pipeline.py` contiene la lógica central del pipeline predictivo y consta de:
1. **Conexión a Base de Datos**:
   - Utiliza SQLAlchemy para conectarse a la base de datos PostgreSQL y extraer datos históricos de ventas e inventario.
2. **Preprocesamiento de Datos**:
   - Limpieza de datos (manejo de valores nulos y outliers).
   - Normalización y transformación de variables.
   - Generación de características temporales (estacionalidad y tendencias).
3. **Entrenamiento de Modelos Predictivos**:
   - Implementa un Random Forest Regressor para estimar la demanda futura de cada SKU.
   - Ajuste de hiperparámetros con GridSearchCV y validación cruzada.
4. **Detección de Anomalías**:
   - Emplea un Isolation Forest para identificar movimientos inusuales en el inventario.
5. **Generación de Recomendaciones**:
   - Analiza las predicciones y el inventario actual para calcular puntos de reorden y cantidades óptimas de pedido mediante un modelo de optimización lineal (PuLP) con la lógica de EOQ.
6. **Almacenamiento de Resultados**:
   - Guarda las predicciones y recomendaciones en la base de datos.
   - Mantiene un historial de predicciones para análisis posterior.

## Estructura
- `docker-compose.yml`: define servicios de base de datos (Postgres) y la aplicación.
- `.env.example`: variables de entorno.
- `Dockerfile`: contenedor de la app Python.
- `requirements.txt`: dependencias.
- `scripts/seed_db.py`: semilla de datos de ejemplo.
- `src/implementacion_pipeline.py`: código del pipeline.

## Pasos para levantar el proyecto

1. Copiar `.env.example` a `.env` y completar credenciales.
2. Ejecutar:
   ```bash
   docker-compose up --build