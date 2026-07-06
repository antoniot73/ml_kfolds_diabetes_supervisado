# Análisis, implementación y evaluación de Random Forest, SVM Lineal y Red Neuronal MLP mediante K-Folds para clasificación de diabetes

**Repositorio:**  
https://github.com/antoniot73/ml_kfolds_diabetes_supervisado

**GitHub Pages:**  
https://antoniot73.github.io/ml_kfolds_diabetes_supervisado/notebooks/practica_kfolds_diabetes_supervisado.html

**Binder:**  
https://mybinder.org/v2/gh/antoniot73/ml_kfolds_diabetes_supervisado/main?filepath=notebooks/practica_kfolds_diabetes_supervisado.ipynb

---

# Instituto Internacional de Aguascalientes

**Maestría en Inteligencia Artificial para la Transformación Digital**  
**Programa:** Aprendizaje Inteligente  
**Alumno:** Antonio Nicolás Toro González  
**Tutor:** Dr. Francisco Javier Luna Rosas

---

# Descripción

Este repositorio contiene la práctica **"Análisis, implementación y prueba de máquinas de aprendizaje supervisado"**, desarrollada mediante tres algoritmos de clasificación supervisada:

- Random Forest
- Máquina de Soporte Vectorial Lineal (SVM Lineal)
- Red Neuronal Multicapa (MLP)

Los modelos se aplican al problema de clasificación binaria del dataset **Pima Indians Diabetes Database**, utilizando **Stratified K-Fold Cross-Validation** como técnica de validación para comparar objetivamente su desempeño.

La práctica integra el desarrollo del marco teórico, implementación de modelos, evaluación experimental y análisis comparativo de resultados mediante un pipeline reproducible.

---

# Objetivo general

Implementar, evaluar y comparar modelos de aprendizaje supervisado mediante **Stratified K-Fold Cross-Validation**, analizando su capacidad para clasificar pacientes con diabetes mediante métricas de desempeño ampliamente utilizadas en problemas de clasificación binaria.

---

# Objetivos específicos

- Analizar el funcionamiento de K-Fold Cross-Validation.
- Comprender el uso de StratifiedKFold.
- Implementar Random Forest, SVM Lineal y Red Neuronal MLP.
- Validar automáticamente el dataset.
- Comparar el desempeño de los modelos mediante las métricas solicitadas.
- Generar tablas y gráficas reproducibles.
- Exportar automáticamente los resultados.

---

# Dataset

**Nombre:** Pima Indians Diabetes Database

**Fuente:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

El conjunto de datos contiene:

- 768 observaciones
- 8 variables predictoras
- 1 variable objetivo binaria

Variable objetivo:

- Outcome = 0 → No diabetes
- Outcome = 1 → Diabetes

Variables predictoras:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

---

# Técnica de validación

La evaluación se realizó mediante **Stratified K-Fold Cross-Validation** con **k = 10**.

En cada iteración:

- un fold se reserva para prueba;
- los nueve folds restantes se utilizan para entrenamiento.

El uso de **StratifiedKFold** conserva aproximadamente la proporción original de clases en cada partición, obteniendo una estimación más estable del desempeño de los modelos.

---

# Tecnologías utilizadas

- Python 3.12
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- pathlib
- logging

---

# Modelos implementados

Se seleccionaron tres algoritmos representativos de distintos enfoques de aprendizaje supervisado.

## Random Forest

Modelo basado en árboles de decisión que combina múltiples clasificadores mediante votación.

Componentes:

- RandomForestClassifier
- class_weight="balanced"

---

## SVM Lineal

Modelo lineal que maximiza el margen entre clases.

Componentes:

- StandardScaler
- LinearSVC
- Pipeline

---

## Red Neuronal MLP

Modelo no lineal compuesto por capas de neuronas artificiales.

Componentes:

- StandardScaler
- MLPClassifier
- Pipeline

---

# Evaluación

La comparación utiliza:

- Precisión Global
- Error Global
- Precisión Positiva (PP)
- Precisión Negativa (PN)
- Falsos Positivos (FP)
- Falsos Negativos (FN)
- Asertividad Positiva (AP)
- Asertividad Negativa (NP)
- Matriz de Confusión

---

# Resultados principales

| Modelo | Precisión Global | Error Global | PP | PN | FP | FN | AP | NP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **0.7552** | **0.2448** | **0.7724** | **0.7460** | **127** | **61** | **0.6198** | **0.8594** |
| SVM Lineal | 0.7487 | 0.2513 | 0.7090 | 0.7700 | 115 | 78 | 0.6230 | 0.8315 |
| Red Neuronal MLP | 0.7031 | 0.2969 | 0.6269 | 0.7440 | 128 | 100 | 0.5676 | 0.7881 |

## Interpretación general

**Random Forest** obtuvo el mejor desempeño general bajo la estrategia **K-Folds**, alcanzando la mayor precisión global, la mayor precisión positiva y el menor número de falsos negativos. **SVM Lineal** presentó un desempeño cercano, mientras que **Red Neuronal MLP** obtuvo el menor rendimiento bajo la configuración utilizada.

---

# Resultados del proyecto

La ejecución automática del pipeline genera:

- Reporte HTML reproducible.
- Notebook ejecutable.
- Predicciones K-Folds.
- Métricas comparativas.
- Matrices de confusión.
- Comparación gráfica de modelos.
- Importancia de variables para Random Forest.

---

# Archivos generados

```
outputs/
├── graficas/
└── tablas/
```

Incluye:

- distribución de clases;
- matrices de confusión;
- comparación de métricas;
- importancia de variables;
- métricas CSV;
- predicciones CSV.

---

# Estructura del repositorio

```
ml_kfolds_diabetes_supervisado/
│
├── data/
├── notebooks/
├── outputs/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Instalación

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución local

```bash
python src/main.py
```

Notebook:

```bash
jupyter notebook
```

HTML:

```bash
jupyter nbconvert --to html --execute notebooks/practica_kfolds_diabetes_supervisado.ipynb --output practica_kfolds_diabetes_supervisado.html
```

---

# Ejecución en Binder

https://mybinder.org/v2/gh/antoniot73/ml_kfolds_diabetes_supervisado/main?filepath=notebooks/practica_kfolds_diabetes_supervisado.ipynb

---

# GitHub Pages

https://antoniot73.github.io/ml_kfolds_diabetes_supervisado/notebooks/practica_kfolds_diabetes_supervisado.html

---

# Reproducibilidad

El proyecto utiliza:

- `random_state = 42`;
- `StratifiedKFold`;
- validación automática del dataset;
- verificación de valores faltantes;
- rutas relativas;
- exportación automática de tablas y gráficas;
- pipelines con escalamiento interno para evitar fuga de datos.

Estas características permiten reproducir completamente los resultados tanto en entorno local como mediante Binder.

---

# Referencias

Breiman, L. (2001). *Random forests*. Machine Learning, 45(1), 5–32.

Géron, A. (2019). *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

Gironés Roig, J., Casas Roma, J., Minguillón Alfonso, J., & Caihuelas Quiles, R. (2017). *Minería de datos: modelos y algoritmos*. Editorial UOC.

Hastie, T., Tibshirani, R., & Friedman, J. (2017). *The Elements of Statistical Learning*. Springer.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.

Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt Publishing.

UCI Machine Learning Repository. *Pima Indians Diabetes Database*. Kaggle.
