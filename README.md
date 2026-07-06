# Análisis, implementación y evaluación de Random Forest, SVM Lineal y Red Neuronal MLP mediante K-Folds para clasificación de diabetes

**Repositorio:** 
https://github.com/antoniot73/ml_kfolds_diabetes_supervisado

**GitHub Pages:** 
https://antoniot73.github.io/ml_kfolds_diabetes_supervisado/notebooks/practica_kfolds_diabetes_supervisado.html  

**Binder:** 
https://mybinder.org/v2/gh/antoniot73/ml_kfolds_diabetes_supervisado/main?filepath=notebooks/practica_kfolds_diabetes_supervisado.ipynb
---

## Instituto Internacional de Aguascalientes

**Maestría en Inteligencia Artificial para la Transformación Digital**  
**Programa:** Aprendizaje Inteligente  
**Alumno:** Antonio Nicolás Toro González  
**Tutor:** Dr. Francisco Javier Luna Rosas

---

## Descripción

Este repositorio contiene la práctica **“Análisis, implementación y prueba de máquinas de aprendizaje supervisado”**, desarrollada con tres modelos de clasificación supervisada:

- **Random Forest**
- **Máquina de Soporte Vectorial Lineal (SVM Lineal)**
- **Red Neuronal Multicapa (MLP)**

Los modelos se aplican al mismo problema de clasificación binaria: predecir si un registro corresponde a resultado negativo o positivo de diabetes a partir del dataset **Pima Indians Diabetes Database**.

La práctica incluye marco teórico, carga del dataset, validación de datos, exploración visual, preparación de variables, implementación de modelos, evaluación mediante **K-Fold Cross-Validation**, matrices de confusión, comparación de métricas e importancia de variables para Random Forest.

---

## Objetivo general

Implementar, evaluar y comparar máquinas de aprendizaje supervisado sobre un dataset de diabetes, utilizando **K-Folds** como técnica de validación y métricas de clasificación solicitadas para la práctica.

---

## Objetivos específicos

- Analizar el funcionamiento de K-Fold Cross-Validation.
- Implementar Random Forest, SVM Lineal y Red Neuronal MLP.
- Cargar y validar un dataset real de clasificación binaria.
- Preparar variables predictoras y variable objetivo.
- Evaluar los modelos bajo la misma estrategia K-Folds.
- Comparar los resultados mediante precisión global, error global, PP, PN, FP, FN, AP y NP.
- Generar un reporte HTML reproducible desde el notebook.

---

## Dataset

**Nombre:** Pima Indians Diabetes Database  
**Fuente:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

El dataset contiene 768 registros, 8 variables predictoras y una variable objetivo binaria.

### Variable objetivo

- `Outcome`
  - `0`: resultado negativo.
  - `1`: resultado positivo.

### Variables predictoras utilizadas

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

---

## Técnica de validación

La evaluación se realizó mediante **K-Fold Cross-Validation** con `k = 10`.

En cada iteración, un fold se reserva como conjunto de prueba y los folds restantes se utilizan para entrenamiento. En esta práctica se usa `StratifiedKFold` para conservar aproximadamente la proporción de clases en cada partición.

---

## Tecnologías utilizadas

- Python 3.12+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- pathlib
- logging

---

## Modelos implementados

### Random Forest

Random Forest combina múltiples árboles de decisión y genera la predicción final mediante votación mayoritaria. En esta práctica se utiliza por su capacidad para capturar relaciones no lineales en datos tabulares.

Componentes utilizados:

- `RandomForestClassifier`
- `class_weight="balanced"`
- importancia de variables

---

### SVM Lineal

La Máquina de Soporte Vectorial Lineal construye una frontera de decisión maximizando el margen entre clases. Debido a que depende de escalas numéricas, se implementa dentro de un pipeline con estandarización.

Componentes utilizados:

- `StandardScaler`
- `LinearSVC`
- `Pipeline`

---

### Red Neuronal MLP

La Red Neuronal Multicapa es un modelo supervisado no lineal compuesto por capas de neuronas artificiales. En esta práctica se utiliza una arquitectura compacta para clasificación tabular.

Componentes utilizados:

- `StandardScaler`
- `MLPClassifier`
- `Pipeline`

---

## Evaluación

Los modelos se comparan con:

- Precisión Global
- Error Global
- Precisión Positiva (PP)
- Precisión Negativa (PN)
- Falsos Positivos (FP)
- Falsos Negativos (FN)
- Asertividad Positiva (AP)
- Asertividad Negativa (NP)
- Matriz de confusión

---

## Resultados principales

| Modelo | Precisión Global | Error Global | PP | PN | FP | FN | AP | NP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM Lineal | 0.7747 | 0.2253 | 0.5746 | 0.8820 | 59 | 114 | 0.7230 | 0.7946 |
| Random Forest | 0.7656 | 0.2344 | 0.7351 | 0.7820 | 109 | 71 | 0.6438 | 0.8463 |
| Red Neuronal MLP | 0.7031 | 0.2969 | 0.6455 | 0.7340 | 133 | 95 | 0.5654 | 0.7944 |

### Interpretación general

SVM Lineal obtuvo la mejor precisión global y el menor error global. Random Forest obtuvo la mayor precisión positiva y el menor número de falsos negativos, lo que lo hace relevante cuando se busca priorizar la detección de pacientes positivos. La Red Neuronal MLP no superó a los otros modelos bajo esta configuración compacta.

---

## Archivos generados

El notebook genera salidas en:

```text
outputs/
├── graficas/
└── tablas/
```

Incluye:

- distribución de clases
- matriz de confusión Random Forest
- matriz de confusión SVM Lineal
- matriz de confusión Red Neuronal MLP
- comparación de métricas K-Folds
- importancia de variables Random Forest
- métricas comparativas en CSV
- predicciones K-Folds en CSV

---

## Estructura del repositorio

```text
ml_kfolds_diabetes_supervisado/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   ├── practica_kfolds_diabetes_supervisado.ipynb
│   └── practica_kfolds_diabetes_supervisado.html
│
├── outputs/
│   ├── graficas/
│   │   ├── comparacion_metricas_kfolds.png
│   │   ├── distribucion_clases_dataset.png
│   │   ├── importancia_variables_kfolds_random_forest.png
│   │   ├── matriz_confusion_kfolds_random_forest.png
│   │   ├── matriz_confusion_kfolds_red_neuronal_mlp.png
│   │   └── matriz_confusion_kfolds_svm_lineal.png
│   │
│   └── tablas/
│       ├── importancia_variables_random_forest.csv
│       ├── metricas_kfolds_modelos.csv
│       └── predicciones_kfolds_cache.csv
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── main.py
│   ├── modeling.py
│   ├── reporting.py
│   └── visualization.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución local

Ejecutar el pipeline modular:

```bash
python src/main.py
```

Abrir Jupyter:

```bash
jupyter notebook
```

Ejecutar:

```bash
notebooks/practica_kfolds_diabetes_supervisado.ipynb
```

Generar HTML:

```bash
jupyter nbconvert --to html --execute notebooks/practica_kfolds_diabetes_supervisado.ipynb --output practica_kfolds_diabetes_supervisado.html
```

---

## Ejecución en Binder

Abrir:

https://mybinder.org/v2/gh/antoniot73/ml_kfolds_diabetes_supervisado/main?filepath=notebooks/practica_kfolds_diabetes_supervisado.ipynb

---

## Publicación en GitHub Pages

El reporte HTML puede consultarse en:

https://antoniot73.github.io/ml_kfolds_diabetes_supervisado/notebooks/practica_kfolds_diabetes_supervisado.html

---

## Reproducibilidad

La práctica utiliza:

- semilla aleatoria fija
- K-Folds estratificado
- validación de columnas
- revisión de valores faltantes
- rutas relativas
- exportación de gráficas y tablas
- pipelines con escalamiento interno para evitar fuga de datos

Esto permite ejecutar el proyecto tanto en entorno local como en cloud.

---

## Referencias

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O’Reilly Media.

Gironés Roig, J., Casas Roma, J., Minguillón Alfonso, J., & Caihuelas Quiles, R. (2017). *Minería de datos: modelos y algoritmos*. Editorial UOC.

Hastie, T., Tibshirani, R., & Friedman, J. (2017). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer.

Raschka, S., & Mirjalili, V. (2019). *Python machine learning* (3rd ed.). Packt Publishing.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: with applications in R*. Springer New York.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

UCI Machine Learning. (2016). *Pima Indians diabetes database* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
