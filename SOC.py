#------------------------------------- CARGAR LIBRERÍAS -----------------------------------------
import datetime
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kstest,chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score, RandomizedSearchCV, KFold,RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor,StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.decomposition import PCA
from optbinning import ContinuousOptimalBinning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout,LSTM,GRU, Bidirectional,Conv1D, MaxPooling1D, Flatten, TimeDistributed,Input,Permute,Reshape,multiply
from tensorflow.keras.optimizers import Adam,SGD,RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2,l1,l1_l2

#-------- FUNCIÓN PARA CARGAR LOS DATOS DE LOS CICLOS DE DESCARGA: https://github.com/bnarms/NASA-Battery-Dataset -----------------------------------------
def disch_data(battery):
  mat = loadmat('C:/Users/augus/OneDrive/Desktop/Máster Big Data/TFM/Dataset/' + battery + '.mat') #get the .mat file
  print('Total data in dataset: ', len(mat[battery][0, 0]['cycle'][0])) #get the length of the data from number of cycles
  c = 0 #set a variable to zero
  disdataset = [] #create an empty list for discharge data
  capacity_data = []
  
  for i in range(len(mat[battery][0, 0]['cycle'][0])):
    row = mat[battery][0, 0]['cycle'][0, i] #get each row of the cycle
    if row['type'][0] == 'discharge': #if the row is a dicharge cycle
      ambient_temperature = row['ambient_temperature'][0][0] #get temp,date_time stamp,capacity,voltage,current etc,.
      date_time = datetime.datetime(
            int(row['time'][0][0]),
            int(row['time'][0][1]),
            int(row['time'][0][2]),
            int(row['time'][0][3]),
            int(row['time'][0][4])
        ) + datetime.timedelta(seconds=int(row['time'][0][5]))
      data = row['data']
      capacity = data[0][0]['Capacity'][0][0]
      for j in range(len(data[0][0]['Voltage_measured'][0])):
        voltage_measured = data[0][0]['Voltage_measured'][0][j]
        current_measured = data[0][0]['Current_measured'][0][j]
        temperature_measured = data[0][0]['Temperature_measured'][0][j]
        current_load = data[0][0]['Current_load'][0][j]
        voltage_load = data[0][0]['Voltage_load'][0][j]
        time = data[0][0]['Time'][0][j]
        disdataset.append([c + 1, ambient_temperature, date_time, capacity,
                        voltage_measured, current_measured,
                        temperature_measured, current_load,
                        voltage_load, time])
        capacity_data.append([c + 1, ambient_temperature, date_time, capacity])
      c = c + 1
  print(disdataset[0])
  return [pd.DataFrame(data=disdataset,
                       columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity', 'voltage_measured',
                                'current_measured', 'temperature_measured',
                                'current', 'voltage', 'time']),
          pd.DataFrame(data=capacity_data,
                       columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity'])]

#-------- FUNCIONES PARA CALCULAR LA V DE CRAMER Y GRAFICARLA (AUTORÍA: PROF.ROSA ESPINOLA DE LA UNIVERSIDAD COMPLUTENSE DE MADRID)-----------------------------------------
def Vcramer(v, target):
    if v.dtype == 'float64' or v.dtype == 'int64':
        # Si v es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(v.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        v = pd.cut(v, bins=p)
        v = v.fillna(v.min())

    if target.dtype == 'float64' or target.dtype == 'int64':
        # Si target es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(target.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        target = pd.cut(target, bins=p)
        target = target.fillna(target.min())

    # Calcula una tabla de contingencia entre v y target
    tabla_cruzada = pd.crosstab(v, target)

    # Calcula el chi-cuadrado y el coeficiente V de Cramer
    chi2 = chi2_contingency(tabla_cruzada)[0]
    n = tabla_cruzada.sum().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabla_cruzada.shape) - 1)))

    return v_cramer

def graficoVcramer(matriz, target):
    # Calcula el coeficiente V de Cramer para cada columna de matriz y target
    salidaVcramer = {x: Vcramer(matriz[x], target) for x in matriz.columns}
    # Ordena los resultados en orden descendente por el coeficiente V de Cramer
    sorted_data = dict(sorted(salidaVcramer.items(), key=lambda item: item[1], reverse=True))
    # Crea el gráfico de barras horizontales
    plt.figure(figsize=(10, 6))
    plt.barh(list(sorted_data.keys()), list(sorted_data.values()), color='skyblue')
    plt.xlabel('V de Cramer')
    plt.show()

#--------------- FUNCIÓN PARA APLICAR EL MECANISMO DE ATENCIÓN: https://github.com/PsiPhiTheta/LSTM-Attention -------------    
def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(1, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

#------------------------------------- EXPLORACION DEL DATASET ----------------------------------------

# Cargamos los datos de la batería B0005 con la función definida anteriormente 
df,capacity = disch_data('B0005')
df=df.drop('datetime',axis=1) # no aporta explicabilidad
df.rename(columns={'voltage_measured': 'voltage_battery','current_measured': 'current_battery',
                       'time': 'cycle_time'}, inplace=True)
df['SOH']=df['capacity']/2 # Creamos la variable SOH para ver si tiene relacion con la objetivo (SOC)
df['current'] = df['current'].abs() # Hay valores de corriente negativos por error de lectura, los pasamos a positivos


# Creamos la columna para el SOC iterando cada ciclo
df['SOC'] = 100.0
for cycle in df['cycle'].unique():
    cycle_df = df[df['cycle'] == cycle]
    # Calculamos el tiempo entre medidas
    cycle_df['delta_time'] = cycle_df['cycle_time'].diff().fillna(0)/ 3600.0
    # Calculamos la suma acumulada de la carga descargada (integral de la corriente por el tiempo)
    cycle_df['charge_removed'] = (cycle_df['current'] * cycle_df['delta_time']).cumsum()
    cycle_df['SOC'] = 100 - (cycle_df['charge_removed'] / (cycle_df['SOH']*2)) * 100
    df.loc[df['cycle'] == cycle, 'SOC'] = cycle_df['SOC']


# Analisamos la distribución de las variables
df_desc = df.describe() # estadisticos basicos de las variables numericas

df = df[df['voltage'] >= 0.01] # filtramos los datos para que no existan periodos de reposo después de completa la descarga

# Graficamos una comparacion de las curvas SOC-tiempo para distintos ciclos
for cycle in [1,50,150]:
  cycle_df = df[df['cycle'] == cycle]
  sns.lineplot(x='cycle_time', y='SOC', data=cycle_df, label=f'Cycle {cycle}', marker='o')
plt.title(f'Battery B0005 SOC curve for different cycles')
plt.xlabel('Cycle Time (s)')
plt.ylabel('SOC (%)')
plt.legend(title='Cycle')
plt.grid(True)
plt.show()

target= 'SOC' #definimos la variable objetivo
# categorizamos el SOH a ver si aporta más explicabilidad
optb = ContinuousOptimalBinning(name='SOH', dtype="numerical")
optb.fit(df['SOH'], df[target])
binning_table = optb.binning_table # creamos una tabla con los rangos creados
binning_table.build()

df['binned_SOH'] = optb.transform(df['SOH'], metric="bins") #añadimos el SOH categorizado al df

inputs = df.columns.tolist() # Obtener los nombres de las columnas del DataFrame
inputs.remove('SOC')
# Obtengo la importancia de las variables
graficoVcramer(df[inputs], df[target])
inputs=['current','voltage','voltage_battery','cycle_time','temperature_measured'] # añado solo las variables con buenos resultados de la V de Cramer

# Correlación entre todas las variables frente a la objetivo 
# Calcular la matriz de correlación de Pearson 
matriz_corr = pd.concat([df[target], df[inputs]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico 
plt.figure(figsize=(8, 6))
sns.set(font_scale=0.9)
sns.heatmap(matriz_corr, annot=False, cmap='coolwarm', fmt=".1f", cbar=True, mask=mask) # Crear un mapa de calor 
plt.title("Matriz de correlación")
plt.show()

#---------------------------------------------- EXPLORACION DE VARIOS MODELOS ML --------------------------------------------

X=df[inputs] #definimos el conjunto de inputs
y=df[target] #definimos el conjunto objetivo
seed= 9 #semilla para reproducibilidad de fenomenos aleatorios
#definimos los procesos de division de datos en la validacion cruzada a utilizar
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
repkfold=RepeatedKFold(n_splits=10,n_repeats=5,random_state=seed)
   
#creamos una lista con los modelos ML a probar   
models = []
models.append(('LR', LinearRegression()))
models.append(('ADA', AdaBoostRegressor(random_state=seed)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('ETR', ExtraTreesRegressor(random_state=seed)))
models.append(('DTR', DecisionTreeRegressor(random_state=seed)))
models.append(('RFR', RandomForestRegressor(random_state=seed)))
models.append(('BAG',BaggingRegressor(random_state= seed)))
models.append(('GB',GradientBoostingRegressor(random_state=seed)))
models.append(('xGB',xgb.XGBRegressor(random_state=seed)))
models.append(('SVR', SVR()))
models.append(('MLP', MLPRegressor(random_state=seed)))

# probamos un pipeline donde se aplica el RobustScaler() a cada modelo y lo evaluamos com validacion cruzada y metrica RMSE
results = []
names = []
rmse_scorer = make_scorer(mean_squared_error, squared=False)
for name, model in models:
    pipeline = Pipeline([('scaler', RobustScaler()),('model', model)])
    results.append(cross_val_score(pipeline,X,y,cv=kfold,scoring=rmse_scorer))
    names.append(name)

# calculamos la media y desviacion tipica de los resultados de cada modelo
model_means = np.mean(results, axis=1)
model_stds = np.std(results, axis=1)
for name, mean, std in zip(names, model_means, model_stds):
    print(f"Model: {name}, RMSE Mean: {mean:.18f}, RMSE Std: {std:.18f}")

# creamos los conjuntos de entrenamiento y test (75% entrenamiento)    
train_data = df[df['cycle'] <= 125]
test_data = df[df['cycle'] > 125]
print(test_data.shape[0]/(train_data.shape[0]+test_data.shape[0]))
x_train= train_data[inputs]
x_test= test_data[inputs]
y_train= train_data[target]
y_test= test_data[target]
    
#------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS ML (CASO 75% DATOS ENTRENAMIENTO) -------------------------------------------

# Definición del modelo
ETR = ExtraTreesRegressor(random_state=seed)
# Definición de la red de búsqueda de hiperparámetros
param_grid_ETR = {
    'model__n_estimators': [50,100,200,300],
    'model__max_depth': [5,10, 20, 30],              
    'model__min_samples_split': [2,5,10],             
    'model__min_samples_leaf': [2,5,10],                
    'model__max_features': ['sqrt', 'log2'],     
    'model__bootstrap': [True, False],                   
}

pipeline = Pipeline([('scaler', RobustScaler()),('model', ETR)]) # pipeline para aplicar el escalador
# inicializamos la busqueda 
random_search_ETR = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_ETR, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_ETR = random_search_ETR.fit(x_train, y_train) # ajustamos la busqueda a los datos de entrenamiento

print("Mejores hiperparámetros encontrados:", grid_model_ETR.best_params_) # obtenemos los hiperparámetros del mejor modelo
print(grid_model_ETR.best_score_) # obtenemos la evaluación del RMSE de entrenamimento del mejor modelo
best_ETR_model = grid_model_ETR.best_estimator_ # creamos un modelo con los mejores hiperparámetros
y_pred_ETR = best_ETR_model.predict(x_test)  # hacemos predicciones en el conjunto de test con el mejor modelo
RMSE = mean_squared_error(y_test, y_pred_ETR, squared=False) # calculamos el RMSE de test
MAE = mean_absolute_error(y_test, y_pred_ETR)  # calculamos el MAE de test
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE) 
    
# repetimos el proceso con otra red, ajustando las listas teniendo en cuenta los mejores hiperparámetros encontrados con la anterior red
param_grid_ETR2 = {
    'model__n_estimators': [75,100,125,150,175],
    'model__max_depth': [1,2,3,4,5],              
    'model__min_samples_split': [6,7,8,9,10],             
    'model__min_samples_leaf': [6,7,8,9,10],                
    'model__max_features': ['sqrt'],     
    'model__bootstrap': [False],                   
}
random_search_ETR2 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_ETR2, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_ETR2 = random_search_ETR2.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_ETR2.best_params_)
print(grid_model_ETR2.best_score_)
best_ETR_model2 = grid_model_ETR2.best_estimator_
y_pred_ETR2 = best_ETR_model2.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_ETR2, squared=False)
MAE= mean_absolute_error(y_test, y_pred_ETR2)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)
    
#--------------------------------------------------- RANDOM FOREST REGRESSOR ----------------------------------------------------
RFR = RandomForestRegressor(random_state=seed)
param_grid_RFR = {
    'model__n_estimators':[50,100,150,200,300,400,500],
    'model__max_depth': [2,5,10,15,20,25,30],              
    'model__min_samples_split': [1,2,5,10,15],             
    'model__min_samples_leaf': [1,2,5,10,15],                
    'model__max_features': ['sqrt', 'log2'],
    'model__max_leaf_nodes': [2,5, 10, 20, 30, 40, 50],
    'model__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
    'model__criterion': ['mse', 'friedman_mse', 'mae', 'poisson']                   
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', RFR)])
random_search_RFR = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_RFR, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_RFR = random_search_RFR.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_RFR.best_params_)
print(grid_model_RFR.best_score_)
best_RFR_model = grid_model_RFR.best_estimator_
y_pred_RFR = best_RFR_model.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_RFR, squared=False)
MAE= mean_absolute_error(y_test, y_pred_RFR)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)


param_grid_RFR2 = {
    'model__n_estimators':[110,120,130,140,150,160,170,180,190],
    'model__max_depth': [16,17,18,19,20,21,22,23,24],              
    'model__min_samples_split': [2,3,4],             
    'model__min_samples_leaf': [3,4,5,6,7,8,9],                
    'model__max_features': ['log2'],
    'model__max_leaf_nodes': [1,2],
    'model__min_impurity_decrease': [0.1,0.2,0.5],
    'model__criterion': ['poisson']                   
}
random_search_RFR2 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_RFR2, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_RFR2 = random_search_RFR2.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_RFR2.best_params_)
print(grid_model_RFR2.best_score_)
best_RFR_model2 = grid_model_RFR2.best_estimator_
y_pred_RFR2 = best_RFR_model2.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_RFR2, squared=False)
MAE= mean_absolute_error(y_test, y_pred_RFR2)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#--------------------------------------------------- BAGGING ----------------------------------------------------
BAG = BaggingRegressor(random_state=seed)
param_grid_BAG = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_samples': [0.5,0.8, 1.0],
    'model__max_features': [0.5, 0.8,1.0],
    'model__bootstrap': [True, False],
    'model__bootstrap_features': [True, False]
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', BAG)])
random_search_BAG = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_BAG, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_BAG = random_search_BAG.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_BAG.best_params_)
print(grid_model_BAG.best_score_)
best_BAG_model = grid_model_BAG.best_estimator_
y_pred_BAG = best_BAG_model.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_BAG, squared=False)
MAE= mean_absolute_error(y_test, y_pred_BAG)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_BAG2 = {
    'model__n_estimators': [150,200,250],
    'model__max_samples': [0.1,0.2,0.3,0.4,0.5],
    'model__max_features': [0.1,0.2,0.3,0.4,0.5],
    'model__bootstrap': [True],
    'model__bootstrap_features': [True]
}
random_search_BAG2 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_BAG2, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_BAG2 = random_search_BAG2.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_BAG2.best_params_)
print(grid_model_BAG2.best_score_)
best_BAG_model2 = grid_model_BAG2.best_estimator_
y_pred_BAG2 = best_BAG_model2.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_BAG2, squared=False)
MAE= mean_absolute_error(y_test, y_pred_BAG2)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_BAG3 = {
    'model__n_estimators': [110,120,130,140,150,160,170,180,190],
    'model__max_samples': [0.1],
    'model__max_features': [0.2],
    'model__bootstrap': [True],
    'model__bootstrap_features': [True]
}
random_search_BAG3 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_BAG3, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_BAG3 = random_search_BAG3.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_BAG3.best_params_)
print(grid_model_BAG3.best_score_)
best_BAG_model3 = grid_model_BAG3.best_estimator_
y_pred_BAG3 = best_BAG_model3.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_BAG3, squared=False)
MAE= mean_absolute_error(y_test, y_pred_BAG3)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS ML (CASO 50% DATOS ENTRENAMIENTO) -------------------------------------------

train_data2 = df[df['cycle'] <= 86]
test_data2 =  df[df['cycle'] > 86]
x_train2= train_data2[inputs]
x_test2= test_data2[inputs]
y_train2= train_data2[target]
y_test2= test_data2[target]

#---------------------------- EXTRA TREES REGRESSOR --------------------------------------
grid_model_ETR4 = random_search_ETR.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_ETR4.best_params_)
print(grid_model_ETR4.best_score_)
# Evaluate on the test set
best_ETR_model4 = grid_model_ETR4.best_estimator_
y_pred_ETR4 = best_ETR_model4.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_ETR4, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_ETR4)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_ETR5 = {
    'model__n_estimators': [60,80,100,120,140,160,180],
    'model__max_depth': [1,2,3,4,5],              
    'model__min_samples_split': [3,4,5,6,7,8,9],             
    'model__min_samples_leaf': [3,4,5,6,7,8,9],                
    'model__max_features': ['log2'],     
    'model__bootstrap': [False],                   
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', ETR)])
random_search_ETR5 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_ETR5, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_ETR5 = random_search_ETR5.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_ETR5.best_params_)
print(grid_model_ETR5.best_score_)
# Evaluate on the test set
best_ETR_model5 = grid_model_ETR5.best_estimator_
y_pred_ETR5 = best_ETR_model5.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_ETR5, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_ETR5)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#-------------------------------------- RANDOM FOREST REGRESSOR --------------------------------
grid_model_RFR3 = random_search_RFR.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_RFR3.best_params_)
print(grid_model_RFR3.best_score_)
best_RFR_model3 = grid_model_RFR3.best_estimator_
y_pred_RFR3 = best_RFR_model3.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_RFR3, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_RFR3)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_RFR4 = {
    'model__n_estimators':[320,340,360,380,400,420,440,460,480],
    'model__max_depth': [21,22,23,24,25,26,27,28,29],              
    'model__min_samples_split': [3,4,5,6,7,8,9],             
    'model__min_samples_leaf': [1],                
    'model__max_features': ['sqrt'],
    'model__max_leaf_nodes': [1,2,3,4],
    'model__criterion': ['poisson']                   
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', RFR)])
random_search_RFR4 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_RFR4, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_RFR4 = random_search_RFR4.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_RFR4.best_params_)
print(grid_model_RFR4.best_score_)
best_RFR_model4 = grid_model_RFR4.best_estimator_
y_pred_RFR4 = best_RFR_model4.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_RFR4, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_RFR4)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#-------------------------------------- BAGGING -----------------------------------------

grid_model_BAG3 = random_search_BAG.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_BAG3.best_params_)
print(grid_model_BAG3.best_score_)
best_BAG_model3 = grid_model_BAG3.best_estimator_
y_pred_BAG3 = best_BAG_model3.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_BAG3, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_BAG3)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_BAG4 = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_samples': [0.5,0.8, 1.0],
    'model__max_features': [0.5, 0.8,1.0],
    'model__bootstrap': [True],
    'model__bootstrap_features': [True]
}

pipeline = Pipeline([('scaler', RobustScaler()),('model', BAG)])
random_search_BAG4 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_BAG4, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_BAG4 = random_search_BAG4.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_BAG4.best_params_)
print(grid_model_BAG4.best_score_)
best_BAG_model4 = grid_model_BAG4.best_estimator_
y_pred_BAG4 = best_BAG_model4.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_BAG4, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_BAG4)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS DL (CASO 75% DATOS ENTRENAMIENTO) -------------------------------------------

scaler = RobustScaler() #Inicializamos el escalador
x_train = scaler.fit_transform(x_train) # transformamos los datos de entrenamiento
x_test = scaler.transform(x_test) # transformamos los datos de test
x_train_array = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])) # redimensionamos los datos para introducir el tamaño de la secuencia temporal (1)
x_test_array = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
x_train2 = scaler.fit_transform(x_train2)
x_test2 = scaler.transform(x_test2)
x_train_array2 = x_train2.reshape((x_train2.shape[0], 1, x_train2.shape[1]))
x_test_array2 = x_test2.reshape((x_test2.shape[0], 1, x_test2.shape[1]))

#----------------------------------------------- RNN GRU ------------------------------------------------------
model_GRU = Sequential() # inicializamos la creación del modelo con Sequential()
# añadimos capas posibles dropouts y el tamaño del input
model_GRU.add(Bidirectional(GRU(units=128, return_sequences=True), input_shape=(1, 5)))
model_GRU.add(Bidirectional(GRU(units=128, return_sequences=False)))
model_GRU.add(Dense(units=1, activation='linear'))  # añadimos capa de salida especificando la función de activación
model_GRU.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error']) # Compilamos seleccionando un otimizador y la función de pérdida
model_GRU.summary() # imprimimos un cuadro-resumen del modelo
callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=10) # definimos el callback de convergencia
history = model_GRU.fit(x_train_array, y_train, batch_size=32, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback]) # ajustamos el modelo a los datos con un máximo de 100 ciclos
# construimos la grafica del historial de evaluacion de los errores del modelo
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_GRU.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_GRU = model_GRU.predict(x_test_array)

#----------------------------------------------- RNN LSTM  -----------------------------------------------------

model_LSTM = Sequential()
model_LSTM.add(LSTM(units=256, return_sequences=True, input_shape=(1, 5)))
model_LSTM.add(LSTM(units=256, return_sequences=True, input_shape=(1, 5)))
model_LSTM.add(Dense(units=1, activation='linear'))
model_LSTM.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_LSTM.summary()
history_LSTM = model_LSTM.fit(x_train_array, y_train, batch_size=64, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback])
plt.plot(history_LSTM.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_LSTM.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_LSTM.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_LSTM = model_LSTM.predict(x_test_array)

#----------------------------------------------- ANN -----------------------------------------------------
model_ANN = Sequential()
model_ANN.add(Dense(units=256, activation='relu', input_shape=(1,5)))  # input_dim should be the number of features in your dataset
model_ANN.add(Dense(units=256, activation='relu'))
model_ANN.add(Dense(units=1, activation='linear'))
model_ANN.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_ANN.summary()
history_ANN = model_ANN.fit(x_train_array, y_train, batch_size=32, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback])
plt.plot(history_ANN.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_ANN.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_ANN.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))

#----------------------------------------------- CNN-GRU  -----------------------------------------------------
model_CNNGRU = Sequential()
model_CNNGRU.add(TimeDistributed(Conv1D(128, 3, activation='relu'),
                          input_shape=(1,5,1)))
model_CNNGRU.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNGRU.add(TimeDistributed(Flatten()))
model_CNNGRU.add(GRU(256,return_sequences=True))
model_CNNGRU.add(Dense(units=1, activation='linear'))
model_CNNGRU.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_CNNGRU.summary()
history_CNNGRU = model_CNNGRU.fit(x_train_array, y_train, batch_size=64, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback])
plt.plot(history_CNNGRU.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_CNNGRU.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_CNNGRU.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_CNNGRU = model_CNNGRU.predict(x_test_array)

#----------------------------------------------- CNN-LSTM  ------------------------------------------------------------------------------------

model_CNNLSTM = Sequential()
model_CNNLSTM.add(TimeDistributed(Conv1D(128, 3, activation='relu'),
                          input_shape=(1,5,1)))
model_CNNLSTM.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNLSTM.add(TimeDistributed(Flatten()))
model_CNNLSTM.add(LSTM(256,return_sequences=True))
model_CNNLSTM.add(LSTM(256,return_sequences=True))
model_CNNLSTM.add(Dense(units=1, activation='relu'))
model_CNNLSTM.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_CNNLSTM.summary()
history_CNNLSTM = model_CNNLSTM.fit(x_train_array, y_train, batch_size=32, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback])
plt.plot(history_CNNLSTM.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_CNNLSTM.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_CNNLSTM.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_CNNLSTM = model_CNNLSTM.predict(x_test_array)

#----------------------------------------------- CNN-LSTM-ATT --------------------------------------------------------------------------------

def cnn_gru_att_after():
    inputs = Input(shape=(1, 5,1))
    out=TimeDistributed(Conv1D(128, 3, activation='relu'))(inputs)
    out=TimeDistributed(MaxPooling1D(pool_size=1))(out)
    out=TimeDistributed(Flatten())(out)
    out=Bidirectional(GRU(256,return_sequences=True))(out)
    attention_mul = attention_3d_block(out)
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

cnn_gru_att=cnn_gru_att_after()
cnn_gru_att.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
print(cnn_gru_att.summary())
history_cnn_gruatt=cnn_gru_att.fit(x_train_array, y_train, epochs=100, batch_size=64, validation_data=(x_test_array, y_test), callbacks=[callback])
plt.plot(history_cnn_gruatt.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_cnn_gruatt.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('cnn_gru_att MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = cnn_gru_att.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_CNNGRUATT = cnn_gru_att.predict(x_test_array)

#----------------------------------------------- RNN GRU ----------------------------------------------------------------------------------------
model_GRU50 = Sequential()
model_GRU50.add(GRU(units=256, return_sequences=True, input_shape=(1, 5)))
model_GRU50.add(GRU(units=256, return_sequences=False))
model_GRU50.add(Dense(units=1, activation='relu'))
model_GRU50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_GRU50.summary()
callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
history_GRU50 = model_GRU50.fit(x_train_array2, y_train2, batch_size=64, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_GRU50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_GRU50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_GRU50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_GRU50 = model_GRU50.predict(x_test_array2)

#----------------------------------------------- RNN LSTM  ------------------------------------------------------------------------
model_LSTM50 = Sequential()
model_LSTM50.add(Bidirectional(LSTM(units=256, return_sequences=True), input_shape=(1, 5)))
model_LSTM50.add(Dense(units=1, activation='linear'))
model_LSTM50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_LSTM50.summary()
history_LSTM50 = model_LSTM50.fit(x_train_array2, y_train2, batch_size=64, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_LSTM50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_LSTM50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_LSTM50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))

#----------------------------------------------- ANN --------------------------------------------------------------------------------
model_ANN50 = Sequential()
model_ANN50.add(Dense(units=128, activation='relu', input_shape=(1,5)))
model_ANN50.add(Dense(units=128, activation='relu'))
model_ANN50.add(Dense(units=1, activation='relu'))
model_ANN50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_ANN50.summary()
history_ANN50 = model_ANN50.fit(x_train_array2, y_train2, batch_size=32, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_ANN50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_ANN50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_ANN50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
#----------------------------------------------- CNN-GRU  -----------------------------------------------------
model_CNNGRU50 = Sequential()
model_CNNGRU50.add(TimeDistributed(Conv1D(256, 3, activation='relu'),
                          input_shape=(1,5,1)))
model_CNNGRU50.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNGRU50.add(TimeDistributed(Flatten()))
model_CNNGRU50.add(GRU(256,return_sequences=True))
model_CNNGRU50.add(GRU(256,return_sequences=True))
model_CNNGRU50.add(Dense(units=1, activation='relu'))
model_CNNGRU50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_CNNGRU50.summary()
history_CNNGRU50 = model_CNNGRU50.fit(x_train_array2, y_train2, batch_size=32, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_CNNGRU50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_CNNGRU50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_CNNGRU50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_CNNGRU500 = model_CNNGRU50.predict(x_test_array2)

#----------------------------------------------- CNN-LSTM  -----------------------------------------------------
model_CNNLSTM50 = Sequential()
model_CNNLSTM50.add(TimeDistributed(Conv1D(128, 3, activation='relu'),
                          input_shape=(1,5,1)))
model_CNNLSTM50.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNLSTM50.add(TimeDistributed(Flatten()))
model_CNNLSTM50.add(Bidirectional(LSTM(256,return_sequences=True)))
model_CNNLSTM50.add(Dense(units=1, activation='relu'))
model_CNNLSTM50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_CNNLSTM50.summary()
history_CNNLSTM50 = model_CNNLSTM50.fit(x_train_array2, y_train2, batch_size=32, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_CNNLSTM50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_CNNLSTM50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_CNNLSTM50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_CNNLSTM50 = model_CNNLSTM50.predict(x_test_array2)

#----------------------------------------------- CNN-LSTM-ATT -----------------------------------------------------

def cnn_lstm_att_after():
    inputs = Input(shape=(1, 5,1))
    out=TimeDistributed(Conv1D(128, 3, activation='relu'))(inputs)
    out=TimeDistributed(MaxPooling1D(pool_size=1))(out)
    out=TimeDistributed(Flatten())(out)
    out=Bidirectional(LSTM(256,return_sequences=True))(out)
    attention_mul = attention_3d_block(out)
    output = Dense(1, activation='relu')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model
cnn_lstm_att=cnn_lstm_att_after()
cnn_lstm_att.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
print(cnn_lstm_att.summary())
history_cnn_lstmatt=cnn_lstm_att.fit(x_train_array2, y_train2, epochs=100, batch_size=32, validation_data=(x_test_array2, y_test2), callbacks=[callback])
plt.plot(history_cnn_lstmatt.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_cnn_lstmatt.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('cnn_lstm_att MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = cnn_lstm_att.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_CNNLSTMATT_50 = cnn_lstm_att.predict(x_test_array2)

#-------------------------- REPRESENTACION DE PREVISIONES Y ERRORES ------------------------------------------------------------------------
# Cremos un diccionario con las predicciones de los mejores modelos para los casos de conjunto de entrenamiento de 75% y 50%
models = {
    'ETR': (np.append(y_train,y_pred_ETR), np.append(y_train2,y_pred_ETR5)),
    'RFR': (np.append(y_train,y_pred_RFR), np.append(y_train2,y_pred_RFR4)),
    'BAG': (np.append(y_train,y_pred_BAG), np.append(y_train2,y_pred_BAG4)),
    'GRU': (np.append(y_train,y_pred_GRU), np.append(y_train2,y_pred_GRU50)),
    'CNN-GRU': (np.append(y_train,y_pred_CNNGRU), np.append(y_train2,y_pred_CNNGRU500)),
    'CNN-LSTM':  (np.append(y_train,y_pred_CNNLSTM), np.append(y_train2,y_pred_CNNLSTM50)),
    'CNN-GRU-ATT':  (np.append(y_train,y_pred_CNNGRUATT),np.append(y_train,y_pred_CNNGRUATT)),
    'CNN-LSTM-ATT':  (np.append(y_train2,y_pred_CNNLSTMATT_50),np.append(y_train2,y_pred_CNNLSTMATT_50))
}

# Iteramos por los modelos para crear las variables con las predicciones
for model_name, (train_75, train_50) in models.items():
   df[f'SOC_{model_name}_75'] = train_75
   df[f'SOC_{model_name}_50'] = train_50

# creamos variables para los errores relativos de prediccion
for col in ['SOC_ETR_75','SOC_ETR_50','SOC_RFR_75','SOC_RFR_50','SOC_BAG_75','SOC_BAG_50','SOC_GRU_75','SOC_GRU_50','SOC_CNN-GRU_75','SOC_CNN-GRU_50','SOC_CNN-LSTM_75','SOC_CNN-LSTM_50','SOC_CNN-GRU-ATT_75','SOC_CNN-LSTM-ATT_50'] :
   df[f'Error(%)_{col}'] = np.abs(df[col] -df['SOC']) /df['SOC'] * 100
   df[f'Error(Abs)_{col}'] = np.abs(df[col] -df['SOC'])

# representamos las predicciones de los modelos ML para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
for cycle in [130,160]:
    df_cycle = df[df['cycle'] == cycle]
    # Plot each SOC column against the cycle_time
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_ETR_75'], label='ETR Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_RFR_75'], label='RFR Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_BAG_75'], label='Bagging Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC'], label='Test SOC',color='black', linewidth=2)
    # Add labels and title
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC (%)')
    plt.title(f'Model SOC Prediction Comparison in cycle {cycle} (75% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()

# representamos las predicciones de los modelos ML para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
for cycle in [90,120]:
    df_cycle = df[df['cycle'] == cycle]
    # Plot each SOC column against the cycle_time
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_ETR_50'], label='ETR Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_RFR_50'], label='RFR Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_BAG_50'], label='Bagging Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC'], label='Test SOC',color='black', linewidth=2)
    # Add labels and title
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC (%)')
    plt.title(f'Model SOC Prediction Comparison in cycle {cycle} (50% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# representamos las predicciones de los modelos DL para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
for cycle in [130,160]:
    df_cycle = df[df['cycle'] == cycle]
    # Plot each SOC column against the cycle_time
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_GRU_75'], label='GRU Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_CNN-GRU_75'], label='CNN-GRU Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_CNN-LSTM_75'], label='CNN-LSTM Predicted SOC')    
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_CNN-GRU-ATT_75'], label='CNN-GRU-ATT Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC'], label='Test SOC',color='black', linewidth=2)
    # Add labels and title
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC (%)')
    plt.title(f'Model SOC Prediction Comparison in cycle {cycle} (75% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()

# representamos las predicciones de los modelos DL para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
for cycle in [90,120]:
    df_cycle = df[df['cycle'] == cycle]
    # Plot each SOC column against the cycle_time
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_GRU_50'], label='GRU Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_CNN-GRU_50'], label='CNN-GRU Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_CNN-LSTM_50'], label='CNN-LSTM Predicted SOC')    
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC_CNN-LSTM-ATT_50'], label='CNN-LSTM-ATT Predicted SOC')
    plt.plot(df_cycle['cycle_time'], df_cycle['SOC'], label='Test SOC',color='black', linewidth=2)
    # Add labels and title
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC (%)')
    plt.title(f'Model SOC Prediction Comparison in cycle {cycle} (50% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()


# representamos los errores de las predicciones de los modelos ML para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
for cycle in [130,160]:
    df_cycle = df[df['cycle'] == cycle]
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_ETR_75'], label='ETR SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_RFR_75'], label='RFR SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_BAG_75'], label='Bagging SOC Prediction Absolute Error')
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC Prediction Absolute Error')
    plt.title(f'Model SOC Prediction Error Comparison in cycle {cycle} (75% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# representamos los errores de las predicciones de los modelos ML para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
for cycle in [90,120]:
    df_cycle = df[df['cycle'] == cycle]
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_ETR_50'], label='ETR SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_RFR_50'], label='RFR SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_BAG_50'], label='Bagging SOC Prediction Absolute Error')
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC Prediction Absolute Error')
    plt.title(f'Model SOC Prediction Error Comparison in cycle {cycle} (50% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# representamos los errores de las predicciones de los modelos DL para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
for cycle in [130,160]:
    df_cycle = df[df['cycle'] == cycle]
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_GRU_75'], label='GRU SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_CNN-GRU_75'], label='CNN-GRU SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_CNN-LSTM_75'], label='CNN-LSTM SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_CNN-GRU-ATT_75'], label='CNN-GRU-ATT SOC Prediction Absolute Error')
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC Prediction Absolute Error')
    plt.title(f'Model SOC Prediction Error Comparison in cycle {cycle} (75% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# representamos los errores de las predicciones de los modelos DL para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
for cycle in [90,120]:
    df_cycle = df[df['cycle'] == cycle]
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_GRU_50'], label='GRU SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_CNN-GRU_50'], label='CNN-GRU SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_CNN-LSTM_50'], label='CNN-LSTM SOC Prediction Absolute Error')
    plt.plot(df_cycle['cycle_time'], df_cycle['Error(Abs)_SOC_CNN-LSTM-ATT_50'], label='CNN-LSTM-ATT SOC Prediction Absolute Error')
    plt.xlabel('Discharge cycle time (s)')
    plt.ylabel('SOC Prediction Absolute Error')
    plt.title(f'Model SOC Prediction Error Comparison in cycle {cycle} (50% train data)')
    plt.legend()
    plt.grid(True)
    plt.show()

