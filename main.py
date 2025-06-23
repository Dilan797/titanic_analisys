import pandas as pd
import numpy as np

#Cargamos conjunto de rendimiento y de pruebas (APRENDIZAJE AUTOMATICO)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
#Metricas para probar nuestro modelo
from sklearn.metrics import accuracy_score, confusion_matrix
#Visualizacion de datos
import matplotlib.pyplot as plt
import seaborn as sns

#Creamos un marco de datos 
#1Leemos el archivo CSV
data = pd.read_csv('tested.csv')
data.info()
print(data.isnull().sum())

#Limpiamos los datos
def fill_missing_ages(df):
    """Rellena Age con la mediana por Pclass; si aún falta, usa la mediana global."""
    mediana_por_clase = df.groupby("Pclass")["Age"].transform("median")
    df["Age"] = df["Age"].fillna(mediana_por_clase)
    df["Age"] = df["Age"].fillna(df["Age"].median())   # por si queda algún NaN

def preprocess_data(df):
    df = df.copy()
    
    # ------------------------------------------
    # 1. Eliminar columnas que no aportan
    # ------------------------------------------
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    
    # ------------------------------------------
    # 2. Rellenar missing values básicos
    # ------------------------------------------
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)        # si no la vas a usar
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    
    fill_missing_ages(df)                              # Edad
    
    # ------------------------------------------
    # 3. Variables nuevas / recodificaciones
    # ------------------------------------------
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"]    = (df["FamilySize"] == 0).astype(int)
    
    # qcut y cut producen NaN si el valor original lo era 👉 rellenamos después
    df["FareBin"] = pd.qcut(
        df["Fare"],
        4,
        labels=False,
        duplicates="drop"          # evita error si hay muchos empates
    )
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 20, 40, 60, np.inf],
        labels=False,
        right=False
    )
    
    # Cualquier NaN residual de las binarizaciones → modo
    df["FareBin"].fillna(df["FareBin"].mode()[0], inplace=True)
    df["AgeBin"].fillna (df["AgeBin"].mode()[0],  inplace=True)
    
    # ------------------------------------------
    # 4. Comprobación final (opcional pero recomendable)
    # ------------------------------------------
    if df.isnull().any().any():
        # Esto te avisará en desarrollo si algo se escapó
        print("⚠️  Aún hay NaNs tras la limpieza:\n", df.isnull().sum())
        df.dropna(inplace=True)  # o lanza un error, según prefieras
    
    return df
    
#Completar edades que faltan
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            # Calculamos la mediana de la edad para cada clase
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    # Aplicamos el mapeo para llenar los valores faltantes de la edad
    df["Age"] = df.apply(lambda row:age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],
                        axis=1)
    
# Preprocesamos los datos
data = preprocess_data(data)
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Dividimos el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#ML PROCESSING
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train) #Escalamos los datos de entrenamiento
X_test = scaler.transform(X_test) #Transformamos los datos de prueba con el mismo escalador 


#Modelo KNN (K-Nearest Neighbors)
def tune_model(X_train, y_train):
    # Definimos los parámetros para la búsqueda en cuadrícula
    param_grid = {
        'n_neighbors': range(1, 21),  # Probar valores de k entre 1 y 20
        'metric': ['euclidean', 'manhattan', "minkowski"], #Verificar si los puntos de datos estan cerca o lejos
        'weights': ['uniform', 'distance']  # Probar diferentes estrategias de ponderación,
        
    }   
    
    model = KNeighborsClassifier()
    #Busqueda de hiperparámetros en cuadrícula
    grid_search = GridSearchCV( model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_
    
best_model = tune_model(X_train, y_train)



def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

