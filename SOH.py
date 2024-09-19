#------------------------------------- CARGAR LIBRERÍAS -----------------------------------------
import datetime
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kstest, chi2_contingency
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout,LSTM,GRU, Bidirectional,BatchNormalization, Conv1D, MaxPooling1D, Flatten, TimeDistributed,Input,Permute,Reshape,multiply
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

# Función para representar los resultados del PCA
def pca_results(results, names, components):
    df2 = pd.DataFrame(zip(results, names, components), columns=['results', 'names', 'components'])    
    sns.catplot(data=df2, kind="bar", x="names", y="results", hue="components", height=5, aspect=15/5)
    plt.ylim((0.0, 0.004))
    plt.show()
   
#--------------- FUNCIÓN PARA APLICAR EL MECANISMO DE ATENCIÓN: https://github.com/PsiPhiTheta/LSTM-Attention -------------    
def attention_3d_block(inputs):
     a = Permute((2, 1))(inputs)
     a = Dense(1, activation='softmax')(a)
     a_probs = Permute((2, 1), name='attention_vec')(a)
     output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
     return output_attention_mul   

#------------------------------------- EXPLORACION DEL DATASET ----------------------------------------

# Cargamos los datos de cada batería con la función definida anteriormente y los almacenamos en un diccionario de dataframes
batteries = ['B0005','B0006','B0007']
dfs = {}
for x in batteries:
    df,capacity = disch_data(x)
    df = df.drop('ambient_temperature', axis=1) # se mantiene constante, no aporta explicabilidad 
    df=df.drop('datetime',axis=1) #no aporta explicabilidad 
    df.rename(columns={'voltage_measured': 'voltage_battery',
                       'current_measured': 'current_battery',
                       'time': 'cycle_time'}, inplace=True)
    df['SOH']=df['capacity']/2 # creamos la variable objetivo
    dfs[x] = df

column_names = dfs['B0005'].columns.tolist() # Obtener los nombres de las columnas del DataFrame

# Creacción de un dataframe que combine todos los dfs
combined_df = pd.DataFrame()
for battery in batteries:
    df = dfs[battery]
    df['battery'] = battery
    combined_df = pd.concat([combined_df, df])
    
# Analisamos la distribución de las variables
dfs_desc = {}
for x in dfs:
    dfs_desc[x]=dfs[x].describe() # estadisticos basicos de las variables 

# Graficamos los scatter plots de voltage y current para cada batería
for x in dfs:
    for column in ['voltage','current']:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=dfs[x]['cycle'], y=dfs[x][column])
        plt.title('Scatter Plot of ' + x +' '+ column)
        plt.xlabel('Discharge Cycle')
        plt.ylabel('Value')
        plt.show()
        
# El ruido/error de lectura es muy bajo para el voltaje pero la corriente debería tener solo valores positivos
for x in dfs:
    dfs[x]['current'] = dfs[x]['current'].abs()

combined_df['current'] = combined_df['current'].abs()

# Graficamos los scatter plots con regresión lineal de la evolución del SOH para cada batería
sns.lmplot(x='cycle', y='SOH', hue='battery', data=combined_df, height=6, aspect=1.5, 
           markers=['o', 's', 'D'], palette='husl', ci=None)
plt.title('SOH Evolution for Multiple Batteries')
plt.xlabel('Discharge cycle')
plt.ylabel('State of Health (SOH)')
plt.grid(True)
plt.show()

# Graficamos una comparacion de las curvas de descarga para distintos ciclos
for battery in batteries:
    plt.figure(figsize=(10, 6))
    df = dfs[battery]
    for cycle in [1,50,150]:
        cycle_df = df[df['cycle'] == cycle]
        sns.lineplot(x='cycle_time', y='voltage_battery', data=cycle_df, label=f'Cycle {cycle}')
    plt.title(f'Voltage-Time Discharge Curves for Battery {battery}')
    plt.xlabel('Cycle Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(title='Cycle')
    plt.grid(True)
    plt.show()
    
# Graficamos una comparacion de la temperatura de las baterías para distintos ciclos
for battery in batteries:
    plt.figure(figsize=(10, 6))
    df = dfs[battery]
    for cycle in [1,50,150]:
        cycle_df = df[df['cycle'] == cycle]
        sns.lineplot(x='cycle_time', y='temperature_measured', data=cycle_df, label=f'Cycle {cycle}')
    plt.title(f'Temperature-Time Discharge Curves for Battery {battery}')
    plt.xlabel('Cycle Time (s)')
    plt.ylabel('Temperature (C)')
    plt.legend(title='Cycle')
    plt.grid(True)
    plt.show()
    
# Graficamos una comparacion de las curvas de temperatura de las baterias para el ciclo 50
plt.figure(figsize=(10, 6))
for battery in batteries:
    df = dfs[battery]
    cycle_df = df[df['cycle'] == 50]
    sns.lineplot(x='cycle_time', y='temperature_measured', data=cycle_df, label=f'{battery}')
plt.title('Temperature-Time Curves for Different Batteries in Cycle 50')
plt.xlabel('Cycle Time (s)')
plt.ylabel('Temperature (C)')
plt.legend(title='Battery')
plt.grid(True)
plt.show()
    
# Graficamos una comparacion de las curvas de descarga de las baterias para el ciclo 50
plt.figure(figsize=(10, 6))
for battery in batteries:
    df = dfs[battery]
    cycle_df = df[df['cycle'] == 50]
    sns.lineplot(x='cycle_time', y='voltage_battery', data=cycle_df, label=f'{battery}')
plt.title('Voltage-Time Discharge Curves for Different Batteries in Cycle 50')
plt.xlabel('Cycle Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(title='Battery')
plt.grid(True)
plt.show()

target= 'SOH' #definimos la variable objetivo
inputs= column_names[:-1]
# Obtengo la importancia de las variables basando en el valor de V de cramer
graficoVcramer(dfs['B0005'][inputs], dfs['B0005'][target])

for x in dfs:
    dfs[x] = dfs[x].drop('capacity',axis=1) # util simplemente para construir la variable objetivo
    dfs[x] = dfs[x].drop('battery',axis=1)  # util simplemente para construir graficas
    
combined_df = combined_df.drop('capacity',axis=1)
inputs.remove('capacity')

# Correlación entre todas las variables frente a la objetivo 
# Calcular la matriz de correlación de Pearson 
matriz_corr = pd.concat([dfs['B0005'][target], dfs['B0005'][inputs]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear el gráfico
plt.figure(figsize=(8, 6))
sns.set(font_scale=0.9)
sns.heatmap(matriz_corr, annot=False, cmap='coolwarm', fmt=".1f", cbar=True, mask=mask) # Crear un mapa de calor 
plt.title("Matriz de correlación")
plt.show()

for x in dfs:
    dfs[x] = dfs[x].drop('current_battery',axis=1) # altamente relacionado con current y bajo Cramer

combined_df = combined_df.drop('current_battery',axis=1)
inputs.remove('current_battery')

#---------------------------------------------- EXPLORACION DE VARIOS MODELOS ML --------------------------------------------
X=dfs['B0005'][inputs] #definimos el conjunto de inputs
y=dfs['B0005'][target] #definimos el conjunto objetivo
seed= 9  #semilla para reproducibilidad de fenomenos aleatorios
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

# test de Kolmogorov-Smirnov para cada variable a ver si es interesante usar un StandardScaler()
KS_results = {}
for variable in dfs['B0005']:
    est, p_valor = kstest(dfs['B0005'][variable], 'norm', args=(dfs['B0005'][variable].mean(), dfs['B0005'][variable].std()))
    KS_results[variable] = {'K-S Statistic': est, 'p-value': p_valor}

# probamos un pipeline donde se aplica el RobustScaler() a cada modelo y lo evaluamos com validacion cruzada y metrica RMSE
results = []
names = []
rmse_scorer = make_scorer(mean_squared_error, squared=False) #creamos la métrica de evaluación RMSE
for name, model in models:
    pipeline = Pipeline([('scaler', RobustScaler()),('model', model)])
    results.append(cross_val_score(pipeline,X,y,cv=kfold,scoring=rmse_scorer))
    names.append(name)

# calculamos la media y desviacion tipica de los resultados de cada modelo
model_means = np.mean(results, axis=1)
model_stds = np.std(results, axis=1)
for name, mean, std in zip(names, model_means, model_stds):
    print(f"Model: {name}, RMSE Mean: {mean:.18f}, RMSE Std: {std:.18f}")

# mismo proceso pero con MinMaxScaler()
results2 = []
names2 = []
for name, model in models:
    pipeline = Pipeline([('scaler', MinMaxScaler()),('model', model)])
    results2.append(cross_val_score(pipeline,X,y,cv=kfold,scoring=rmse_scorer))
    names2.append(name)
    
model_means2 = np.mean(results2, axis=1)
model_stds2 = np.std(results2, axis=1)
for name, mean, std in zip(names2, model_means2, model_stds2):
    print(f"Model: {name}, RMSE Mean: {mean:.15f}, RMSE Std: {std:.15f}")
    
# creamos los conjuntos de entrenamiento y test (75% entrenamiento)
train_data = dfs['B0005'][dfs['B0005']['cycle'] <= 125]
test_data = dfs['B0005'][dfs['B0005']['cycle'] > 125]
print(test_data.shape[0]/(train_data.shape[0]+test_data.shape[0]))
x_train= train_data[inputs]
x_test= test_data[inputs]
y_train= train_data[target]
y_test= test_data[target]

# creamos nueva lista de modelos ML, ahora con los que han obtenido mejores resultados anteriormente
models2 = []
models2.append(('ETR', ExtraTreesRegressor(random_state=seed)))
models2.append(('DTR', DecisionTreeRegressor(random_state=seed)))
models2.append(('RFR', RandomForestRegressor(random_state=seed)))
models2.append(('BAG',BaggingRegressor(random_state= seed)))
models2.append(('xGB',xgb.XGBRegressor(random_state=seed)))

# probamos un PCA de 2 a 6 componentes con los mejores modelos, aplicando un RobustScaler()
results_PCA = []
names_PCA = []
components = []
for name, model in models2: #verificar esta linha
    for n in range(2,7):
        pipeline = Pipeline([('PCA', PCA(n_components=n)),('scaler', RobustScaler()),('model', model)])
        scores=cross_val_score(pipeline,X,y,cv=kfold,scoring=rmse_scorer)
        results_PCA.append(scores.mean())
        names_PCA.append(name)
        components.append(n)               
        pipeline.fit(x_train, y_train)
pca_results(results_PCA, names_PCA, components) #construimos una grafica con los resultados

#------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS ML (CASO 75% DATOS ENTRENAMIENTO) -------------------------------------------

#--------------------------------------------------- EXTRA TREES REGRESSOR -------------------------------------------------------------------

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
y_pred_ETR = best_ETR_model.predict(x_test) # hacemos predicciones en el conjunto de test con el mejor modelo
RMSE = mean_squared_error(y_test, y_pred_ETR, squared=False) # calculamos el RMSE de test
MAE = mean_absolute_error(y_test, y_pred_ETR) # calculamos el MAE de test
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

# repetimos el proceso con otra red, ajustando las listas teniendo en cuenta los mejores hiperparámetros encontrados con la anterior red
param_grid_ETR2 = {
    'model__n_estimators': [75,100,125,150],
    'model__max_depth': [1,2,3,4,5],              
    'model__min_samples_split': [6,7,8,9,10],             
    'model__min_samples_leaf': [6,7,8,9,10],                
    'model__max_features': ['log2'],     
    'model__bootstrap': [False],                   
}
random_search_ETR2 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_ETR2, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_ETR2 = random_search_ETR2.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_ETR2.best_params_)
print(grid_model_ETR2.best_score_)
best_ETR_model2 = grid_model_ETR2.best_estimator_
y_pred_ETR2 = best_ETR_model2.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_ETR2 , squared=False)
MAE= mean_absolute_error(y_test, y_pred_ETR2 )
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

# probamos con el PCA
pipeline = Pipeline([('PCA', PCA(n_components=3)),('scaler', RobustScaler()),('model', ETR)])
random_search_ETR3 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_ETR, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_ETR3 = random_search_ETR3.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_ETR3.best_params_)
print(grid_model_ETR3.best_score_)
best_ETR_model3 = grid_model_ETR3.best_estimator_
y_pred_ETR3  = best_ETR_model3.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_ETR3 , squared=False)
MAE= mean_absolute_error(y_test, y_pred_ETR3)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#--------------------------------------------------- DECISION TREE REGRESSOR ----------------------------------------------------
DTR = DecisionTreeRegressor(random_state=seed)
param_grid_DTR = {
    'model__max_depth': [2,5,10,15,20,25,30],              
    'model__min_samples_split': [1,2,5,10,15],             
    'model__min_samples_leaf': [1,2,5,10,15],                
    'model__max_features': ['sqrt', 'log2'],
    'model__max_leaf_nodes': [2,5, 10, 20, 30, 40, 50],
    'model__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
    'model__criterion': ['mse', 'friedman_mse', 'mae', 'poisson']                   
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', DTR)])
random_search_DTR = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_DTR, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_DTR = random_search_DTR.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_DTR.best_params_)
print(grid_model_DTR.best_score_)
best_DTR_model = grid_model_DTR.best_estimator_
y_pred_DTR = best_DTR_model.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_DTR, squared=False)
MAE= mean_absolute_error(y_test, y_pred_DTR)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)


param_grid_DTR2 = {
    'model__max_depth': [6,7,8,9,10,11,12,13,14],              
    'model__min_samples_split': [3,4,5,6,7,8,9],             
    'model__min_samples_leaf': [1],                
    'model__max_features': ['log2'],
    'model__max_leaf_nodes': [35, 40, 45],
    'model__min_impurity_decrease': [0.05],
    'model__criterion': ['poisson']                   
}
random_search_DTR2 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_DTR2, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_DTR2 = random_search_DTR2.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_DTR2.best_params_)
print(grid_model_DTR2.best_score_)
best_DTR_model2 = grid_model_DTR2.best_estimator_
y_pred_DTR2 = best_DTR_model2.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_DTR2, squared=False)
MAE= mean_absolute_error(y_test, y_pred_DTR2)
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
    'model__n_estimators':[75,100,125],
    'model__max_depth': [21,22,23,24,25,26,27,28,29],              
    'model__min_samples_split': [6,7,8,9,10,11,12,13,14],             
    'model__min_samples_leaf': [3,4,5,6,7,8,9],                
    'model__max_features': ['log2'],
    'model__max_leaf_nodes': [3,4,5,6,7,8,9],
    'model__min_impurity_decrease': [0.05],
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
    'model__n_estimators': [10, 50, 100, 200, 300,400,500],
    'model__max_samples': [0.5, 0.6,0.7, 0.8, 0.9,1.0],
    'model__max_features': [0.5, 0.6,0.7, 0.8, 0.9,1.0],
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
    'model__n_estimators': [75, 100,125,150,175],
    'model__max_samples': [0.1,0.2,0.3,0.4,0.5],
    'model__max_features': [0.1,0.2,0.3,0.4,0.5],
    'model__bootstrap': [False],
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


#--------------------------------------------------- XGBOOST ----------------------------------------------------
xGB = xgb.XGBRegressor(random_state=seed)
param_grid_xGB = {
    'model__n_estimators': [50,100, 200, 300,400,500],
    'model__learning_rate': [0.01, 0.05, 0.1,],
    'model__max_depth': [2,5,10,15,20,30],
    'model__min_child_weight': [1, 3, 5, 7,10],
    'model__subsample': [0.5, 0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.5, 0.7, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'model__lambda': [0, 1, 5, 10],
    'model__alpha': [0, 0.1, 0.5, 1.0, 5.0]
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', xGB)])
random_search_xGB = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_xGB, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_xGB = random_search_xGB.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_xGB.best_params_)
print(grid_model_xGB.best_score_)
best_xGB_model = grid_model_xGB.best_estimator_
y_pred_xGB = best_xGB_model.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_xGB, squared=False)
MAE= mean_absolute_error(y_test, y_pred_xGB)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)


param_grid_xGB2 = {
    'model__n_estimators': [75,100,125,150],
    'model__learning_rate': [0.01,0.005],
    'model__max_depth': [30,50,70,100],
    'model__min_child_weight': [10,20,30,40,50],
    'model__subsample': [0.8,0.9],
    'model__colsample_bytree': [0.6,0.7],
    'model__gamma': [0.2],
    'model__lambda': [2,3,4,5,6,7,8,9],
    'model__alpha': [0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9]
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', xGB)])
random_search_xGB2 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_xGB2, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_xGB2 = random_search_xGB2.fit(x_train, y_train)
print("Mejores hiperparámetros encontrados:", grid_model_xGB2.best_params_)
print(grid_model_xGB2.best_score_)
best_xGB_model2 = grid_model_xGB2.best_estimator_
y_pred_xGB2 = best_xGB_model2.predict(x_test)
RMSE = mean_squared_error(y_test, y_pred_xGB2, squared=False)
MAE= mean_absolute_error(y_test, y_pred_xGB2)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)


#---------------------------- STACKING ------------------------------------------------------------
estimators = [('ETR', best_ETR_model),('BAG', best_BAG_model)] # definimos los estimadores base
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression()) # construimos el modelo de stacking
pipeline = Pipeline([('scaler', RobustScaler()),('model', stack)])
stack_model = pipeline.fit(x_train, y_train)
y_pred_stack = stack_model.predict(x_test)
y_pred_stack_train = stack_model.predict(x_train)
RMSE_train = mean_squared_error(y_train, y_pred_stack_train, squared=False)
RMSE = mean_squared_error(y_test, y_pred_stack, squared=False)
MAE= mean_absolute_error(y_test, y_pred_stack)
print("Train Set RMSE:", RMSE_train)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)


#------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS ML (CASO 50% DATOS ENTRENAMIENTO) -------------------------------------------
train_data2 = dfs['B0005'][dfs['B0005']['cycle'] <= 86]
test_data2 = dfs['B0005'][dfs['B0005']['cycle'] > 86]
print(test_data2.shape[0]/(train_data2.shape[0]+test_data2.shape[0]))
x_train2= train_data2[inputs]
x_test2= test_data2[inputs]
y_train2= train_data2[target]
y_test2= test_data2[target]

#---------------------------- EXTRA TREES REGRESSOR -----------------------------------------------------------
grid_model_ETR4 = random_search_ETR.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_ETR4.best_params_)
print(grid_model_ETR4.best_score_)
best_ETR_model4 = grid_model_ETR4.best_estimator_
y_pred_ETR4 = best_ETR_model4.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_ETR4 , squared=False)
MAE= mean_absolute_error(y_test2, y_pred_ETR4 )
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)


param_grid_ETR5 = {
    'model__n_estimators': [150,200,250],
    'model__max_depth': [1,2,3,4,5],              
    'model__min_samples_split': [3,4,5,6,7,8,9],             
    'model__min_samples_leaf': [10,15,20,25,30,40],                
    'model__max_features': ['log2'],     
    'model__bootstrap': [True],                   
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', ETR)])
random_search_ETR5 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_ETR5, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_ETR5 = random_search_ETR5.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_ETR5.best_params_)
print(grid_model_ETR5.best_score_)
best_ETR_model5 = grid_model_ETR5.best_estimator_
y_pred_ETR5 = best_ETR_model5.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_ETR5, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_ETR5)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#-------------------------------------- DECISION TREE REGRESSOR -----------------------------------------------------------------------------
grid_model_DTR3 = random_search_DTR.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_DTR3.best_params_)
print(grid_model_DTR3.best_score_)
best_DTR_model3 = grid_model_DTR3.best_estimator_
y_pred_DTR3 = best_DTR_model3.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_DTR3, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_DTR3)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_DTR4 = {
    'model__max_depth': [30,40,50],              
    'model__min_samples_split': [3,4,5,6,7,8,9],             
    'model__min_samples_leaf': [3,4,5,6,7,8,9],                
    'model__max_features': ['log2'],
    'model__max_leaf_nodes': [22,24,26,28,30,32,34,36,38],
    'model__min_impurity_decrease': [0.1,0.2,0.5,0.8],
    'model__criterion': ['poisson']                   
}
pipeline = Pipeline([('scaler', RobustScaler()),('model', DTR)])
random_search_DTR4 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_DTR4, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_DTR4 = random_search_DTR.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_DTR4.best_params_)
print(grid_model_DTR4.best_score_)
best_DTR_model4 = grid_model_DTR4.best_estimator_
y_pred_DTR4 = best_DTR_model4.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_DTR4, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_DTR4)
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
    'model__n_estimators':[500,600,700],
    'model__max_depth': [2,5,10,15,20,25,30],              
    'model__min_samples_split': [15,20,25,30,40],             
    'model__min_samples_leaf': [3,4,5,6,7,8,9],                
    'model__max_features': ['log2'],
    'model__max_leaf_nodes': [3,4,5,6,7,8,9],
    'model__min_impurity_decrease': [0.1,0.2,0.5,0.8],
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

#-------------------------------------- BAGGING ------------------------------------------------------
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
    'model__n_estimators': [1,2,5,10],
    'model__max_samples': [0.8,],
    'model__max_features': [0.6],
    'model__bootstrap': [False],
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

#-------------------------------------- xGB --------------------------------------------------------------------
grid_model_xGB3 = random_search_xGB.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_xGB3.best_params_)
print(grid_model_xGB3.best_score_)
best_xGB_model3 = grid_model_xGB3.best_estimator_
y_pred_xGB3 = best_xGB_model3.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_xGB3, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_xGB3)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

param_grid_xGB4 = {
    'model__n_estimators': [2,10,20,30,40,50],
    'model__learning_rate': [0.01, 0.005],
    'model__max_depth': [6,7,8,9,10,111,12,13,14],
    'model__min_child_weight': [2,3],
    'model__subsample': [0.2,0.3,0.4,0.5],
    'model__colsample_bytree': [1.0],
    'model__gamma': [0.1],
    'model__lambda': [10,12,15,20],
    'model__alpha': [0.8, 1.0,1.5,2.0,3.0]
}

pipeline = Pipeline([('scaler', RobustScaler()),('model', xGB)])
random_search_xGB4 = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid_xGB4, cv=repkfold, n_jobs= -1,scoring=rmse_scorer)
grid_model_xGB4 = random_search_xGB4.fit(x_train2, y_train2)
print("Mejores hiperparámetros encontrados:", grid_model_xGB4.best_params_)
print(grid_model_xGB4.best_score_)
best_xGB_model4 = grid_model_xGB4.best_estimator_
y_pred_xGB4 = best_xGB_model4.predict(x_test2)
RMSE = mean_squared_error(y_test2, y_pred_xGB4, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_xGB4)
print("Test Set RMSE:", RMSE)
print("Test Set MAE:", MAE)

#---------------------------- STACKING ------------------------------------------------------------
estimators = [('ETR', best_ETR_model4),('BAG', best_BAG_model3)]
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
pipeline = Pipeline([('scaler', RobustScaler()),('model', stack)])
stack_model = pipeline.fit(x_train2, y_train2)
y_pred_stack2 = stack_model.predict(x_test2)
y_pred_stack_train2 = stack_model.predict(x_train2)
RMSE_train = mean_squared_error(y_train2, y_pred_stack_train2, squared=False)
RMSE = mean_squared_error(y_test2, y_pred_stack2, squared=False)
MAE= mean_absolute_error(y_test2, y_pred_stack2)
print("Train Set RMSE:", RMSE_train)
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

#----------------------------------------------- RNN GRU ------------------------------------------------------------------------------------------

model_GRU = Sequential() # inicializamos la creación del modelo con Sequential()
# añadimos capas posibles dropouts y el tamaño del input
model_GRU.add(GRU(units=128, return_sequences=True, input_shape=(1, 6)))
model_GRU.add(GRU(units=128, return_sequences=True))
model_GRU.add(Dropout(0.5))
model_GRU.add(Dense(units=1, activation='linear')) #añadimos capa de salida especificando la función de activación
model_GRU.compile(optimizer='RMSprop', loss='mean_absolute_error', metrics=['mean_absolute_error']) # Compilamos seleccionando un otimizador y la función de pérdida
model_GRU.summary() # imprimimos un cuadro-resumen del modelo
callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=10) # definimos el callback de convergencia
history = model_GRU.fit(x_train_array, y_train, batch_size=64, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback]) # ajustamos el modelo a los datos con un máximo de 100 ciclos
# construimos la grafica del historial de evaluacion de los errores del modelo
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_GRU.evaluate(x_test_array, y_test, verbose=1) 
print('Test Loss: {}'.format(results)) #imprimimos los resultados finales del conjunto test

#----------------------------------------------- RNN LSTM  ---------------------------------------------------------------------------------------

model_LSTM = Sequential()
model_LSTM.add(LSTM(units=256, return_sequences=True, input_shape=(1, 6)))
model_LSTM.add(Dense(units=1, activation='relu'))
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

#----------------------------------------------- ANN ---------------------------------------------------------------------------------------

model_ANN = Sequential()
model_ANN.add(Dense(units=256, activation='relu', input_shape=(1,6)))  # input_dim should be the number of features in your dataset
model_ANN.add(Dropout(0.2))
model_ANN.add(Dense(units=256, activation='relu'))
model_ANN.add(Dropout(0.2))
model_ANN.add(Dense(units=1, activation='relu'))
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
y_pred_ANN = model_ANN.predict(x_test_array)

#----------------------------------------------- CNN-LSTM  -----------------------------------------------------------------------------------------------
model_CNNLSTM = Sequential()
model_CNNLSTM.add(TimeDistributed(Conv1D(128, 2, activation='relu'),
                          input_shape=(1,6,1)))
model_CNNLSTM.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNLSTM.add(TimeDistributed(Flatten()))
model_CNNLSTM.add(LSTM(256,return_sequences=True))
model_CNNLSTM.add(LSTM(256,return_sequences=False))
model_CNNLSTM.add(Dense(units=1, activation='sigmoid'))
model_CNNLSTM.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_CNNLSTM.summary()
history_CNNLSTM = model_CNNLSTM.fit(x_train_array, y_train, batch_size=64, epochs=100, validation_data=(x_test_array, y_test),callbacks=[callback])
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

#----------------------------------------------- CNN-LSTM-ATT --------------------------------------------------------------------------------------------
# en los modelos con mecanismo de atención, para facilitar, no utilizamos sequential()
def cnn_lstm_att_after():
    inputs = Input(shape=(1, 6,1))
    out=TimeDistributed(Conv1D(128, 2, activation='relu'))(inputs)
    out=TimeDistributed(MaxPooling1D(pool_size=1))(out)
    out=TimeDistributed(Flatten())(out)
    out=LSTM(256,return_sequences=True)(out)
    out=LSTM(256,return_sequences=True)(out)
    attention_mul = attention_3d_block(out)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

cnn_lstm_att=cnn_lstm_att_after()
cnn_lstm_att.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
print(cnn_lstm_att.summary())
history_cnn_lstmatt=cnn_lstm_att.fit(x_train_array, y_train, epochs=100, batch_size=64, validation_data=(x_test_array, y_test), callbacks=[callback])
plt.plot(history_cnn_lstmatt.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_cnn_lstmatt.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('cnn_lstm_att MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = cnn_lstm_att.evaluate(x_test_array, y_test, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_cnnlstmatt = cnn_lstm_att.predict(x_test_array)

#------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS DL (CASO 50% DATOS ENTRENAMIENTO) -----------------------------------------------------

#----------------------------------------------- RNN GRU ------------------------------------------------------------------------------------------------

model_GRU50 = Sequential()
model_GRU50.add(GRU(units=256, return_sequences=True, input_shape=(1, 6)))
model_GRU50.add(Dropout(0.2))
model_GRU50.add(GRU(units=256, return_sequences=True))
model_GRU50.add(Dropout(0.2))
model_GRU50.add(Dense(units=1, activation='linear'))
model_GRU50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_GRU50.summary()
history_GRU50 = model_GRU50.fit(x_train_array2, y_train2, batch_size=32, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_GRU50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_GRU50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_GRU50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))

#----------------------------------------------- RNN LSTM  -----------------------------------------------------------------------------------------------------

model_LSTM50 = Sequential()
model_LSTM50.add(LSTM(units=256, return_sequences=False, input_shape=(1, 6)))
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
y_pred_LSTM50 = model_LSTM50.predict(x_test_array2)

#----------------------------------------------- ANN ------------------------------------------------------------------------------------------------------------

model_ANN50 = Sequential()
model_ANN50.add(Dense(units=256, activation='relu', input_shape=(1,6)))
model_ANN50.add(Dropout(0.2))
model_ANN50.add(Dense(units=256, activation='relu'))
model_ANN50.add(Dropout(0.2))
model_ANN50.add(Dense(units=1, activation='linear'))
model_ANN50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_ANN50.summary()
history_ANN50 = model_ANN50.fit(x_train_array2, y_train2, batch_size=64, epochs=100, validation_data=(x_test_array2, y_test2),callbacks=[callback])
plt.plot(history_ANN50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_ANN50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_ANN50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_ANN50 = model_ANN50.predict(x_test_array2)

#----------------------------------------------- CNN-LSTM  ---------------------------------------------------------------------------------------------------------

model_CNNLSTM50 = Sequential()
model_CNNLSTM50.add(TimeDistributed(Conv1D(128, 3, activation='relu'),
                          input_shape=(1,6,1)))
model_CNNLSTM50.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNLSTM50.add(TimeDistributed(Flatten()))
model_CNNLSTM50.add(LSTM(256,return_sequences=True))
model_CNNLSTM50.add(LSTM(256,return_sequences=True))
model_CNNLSTM50.add(Dense(units=1, activation='sigmoid'))
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

#----------------------------------------------- CNN-LSTM-ATT ------------------------------------------------------------------------------------------------------

def cnn_lstm_att50_after():
    inputs = Input(shape=(1, 6,1))
    out=TimeDistributed(Conv1D(128, 3, activation='relu'))(inputs)
    out=TimeDistributed(MaxPooling1D(pool_size=1))(out)
    out=TimeDistributed(Flatten())(out)
    out=Bidirectional(LSTM(256,return_sequences=True))(out)
    out=Bidirectional(LSTM(256,return_sequences=True))(out)
    attention_mul = attention_3d_block(out)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model
cnn_lstm_att50=cnn_lstm_att50_after()
cnn_lstm_att50.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
print(cnn_lstm_att50.summary())
history_cnn_lstmatt50=cnn_lstm_att50.fit(x_train_array2, y_train2, epochs=100, batch_size=32, validation_data=(x_test_array2, y_test2), callbacks=[callback])
plt.plot(history_cnn_lstmatt50.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_cnn_lstmatt50.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('cnn_lstm_att MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = cnn_lstm_att50.evaluate(x_test_array2, y_test2, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_cnn_lstm_att_50 = cnn_lstm_att50.predict(x_test_array2)

#-------------------------- REPRESENTACION DE PREVISIONES Y ERRORES ----------------------------------------------------------------------------------------------

# Cremos un diccionario con las predicciones de los mejores modelos para los casos de conjunto de entrenamiento de 75% y 50%
models = {
    'ETR': (np.append(y_train,y_pred_ETR), np.append(y_train2,y_pred_ETR4)),
    'BAG':  (np.append(y_train,y_pred_BAG), np.append(y_train2,y_pred_BAG3)),
    'STACK':  (np.append(y_train,y_pred_stack), np.append(y_train2,y_pred_stack2)),
    'LSTM':  (np.append(y_train,y_pred_LSTM), np.append(y_train2,y_pred_LSTM50)),
    'ANN':  (np.append(y_train,y_pred_ANN), np.append(y_train2,y_pred_ANN50)),
    'CNN-LSTM':  (np.append(y_train,y_pred_LSTM), np.append(y_train2,y_pred_LSTM50)),
    'CNN-LSTM-ATT':  (np.append(y_train,y_pred_cnnlstmatt), np.append(y_train2,y_pred_CNNLSTM50))
}

# Iteramos por los modelos para crear las variables con las predicciones
for model_name, (train_75, train_50) in models.items():
    # Create SOH variables for 75 and 50
    dfs['B0005'][f'SOH_{model_name}_75'] = train_75
    dfs['B0005'][f'SOH_{model_name}_50'] = train_50

# creamos variables para los errores relativos de prediccion
for col in ['SOH_ETR_75','SOH_ETR_50','SOH_BAG_75','SOH_BAG_50','SOH_STACK_75','SOH_STACK_50','SOH_LSTM_75','SOH_LSTM_50','SOH_ANN_75','SOH_ANN_50','SOH_CNN-LSTM_75','SOH_CNN-LSTM_50','SOH_CNN-LSTM-ATT_75','SOH_CNN-LSTM-ATT_50'] :
    dfs['B0005'][f'Error(%)_{col}'] = np.abs(dfs['B0005'][col] - dfs['B0005']['SOH']) / dfs['B0005']['SOH'] * 100


df_cycle_125 = dfs['B0005'].query('cycle > 125') # creamos un dataframe para los ciclos de validacion
df_cycle_125 = df_cycle_125.groupby('cycle').mean().reset_index() # agrupamos por la columna ciclo haciendo una media de las variables para que se puedan representar las predicciones

# lo mismo para el caso de conjunto de entrenamiento de 50%
df_cycle_86 = dfs['B0005'].query('cycle > 86')
df_cycle_86 = df_cycle_86.groupby('cycle').mean().reset_index()

# representamos las predicciones de los modelos ML para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_ETR_75'], label='ETR Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_BAG_75'], label='Bagging Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_STACK_75'], label='Stacking Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH'], label='Test SOH',color='black', linewidth=2)
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH')
plt.title('Model Prediction Comparison (75% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos las predicciones de los modelos ML para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_ETR_50'], label='ETR Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_BAG_50'], label='Bagging Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_STACK_50'], label='Stacking Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH'], label='Test SOH',color='black', linewidth=2)
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH')
plt.title('Model Prediction Comparison (50% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos los errores relativos de prediccion de los modelos ML para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_ETR_75'], label='ETR Prediction Error')
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_BAG_75'], label='Bagging Prediction Error')
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_STACK_75'], label='Stacking Prediction Error')
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH Prediction Error (%)')
plt.title('Model Prediction Error Comparison (75% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos los errores relativos de prediccion de los modelos ML para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_ETR_50'], label='ETR Prediction Error')
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_BAG_50'], label='Bagging Prediction Error')
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_STACK_50'], label='Stacking Prediction Error')
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH Prediction Error (%)')
plt.title('Model Prediction Error Comparison (50% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos las predicciones de los modelos DL para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_ANN_75'], label='ANN Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_LSTM_75'], label='LSTM Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_CNN-LSTM_75'], label='CNN-LSTM Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH_CNN-LSTM-ATT_75'], label='CNN-LSTM-ATT Predicted SOH')
plt.plot(df_cycle_125['cycle'], df_cycle_125['SOH'], label='Test SOH',color='black', linewidth=2)
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH')
plt.title('Model Prediction Comparison (75% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos las predicciones de los modelos DL para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_LSTM_50'], label='LSTM Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_ANN_50'], label='ANN Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_CNN-LSTM_50'], label='CNN-LSTM Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH_CNN-LSTM-ATT_50'], label='CNN-LSTM-ATT Predicted SOH')
plt.plot(df_cycle_86['cycle'], df_cycle_86['SOH'], label='Test SOH',color='black', linewidth=2)
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH')
plt.title('Model Prediction Comparison (50% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos los errores de prediccion de los modelos DL para conjunto de entrenamiento 75%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_LSTM_75'], label='LSTM Prediction Error')
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_ANN_75'], label='ANN Prediction Error')
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_CNN-LSTM_75'], label='CNN-LSTM Prediction Error')
plt.plot(df_cycle_125['cycle'], df_cycle_125['Error(%)_SOH_CNN-LSTM-ATT_75'], label='CNN-LSTM-ATT Prediction Error')
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH Prediction Error (%)')
plt.title('Model Prediction Error Comparison (75% train data)')
plt.legend()
plt.grid(True)
plt.show()

# representamos los errores de prediccion de los modelos DL para conjunto de entrenamiento 50%
plt.figure(figsize=(10, 6))
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_LSTM_50'], label='LSTM Prediction Error')
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_ANN_50'], label='ANN Prediction Error')
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_CNN-LSTM_50'], label='CNN-LSTM Prediction Error')
plt.plot(df_cycle_86['cycle'], df_cycle_86['Error(%)_SOH_CNN-LSTM-ATT_50'], label='CNN-LSTM-ATT Prediction Error')
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH Prediction Error (%)')
plt.title('Model Prediction Error Comparison (50% train data)')
plt.legend()
plt.grid(True)
plt.show()

#------------------------ OPTIMIZACIÓN DE HIPERPARÁMETROS DE LOS MODELOS DL (CASO CICLO DE VIDA ENTERO) --------------------------------------------------------------

# creamos una columna con el $SOH$ inicial de cada batería para evitar un gran error al inicio del ciclo que se propagaria a otros ciclos
initial_soh = {'B0005': dfs['B0005']['SOH'].iloc[0],
               'B0006': dfs['B0006']['SOH'].iloc[0],
               'B0007': dfs['B0007']['SOH'].iloc[0]}
combined_df['initial_SOH'] = df['battery'].map(initial_soh)

train_data3 = combined_df[combined_df['battery'].isin(['B0006', 'B0007'])] # definimos los datos de las baterías B0006 y B0007 como datos de entrenamiento
test_data3 = combined_df[combined_df['battery'].isin(['B0005'])] # definimos los datos de la batería B0005 como datos de validacion
inputs.append('initial_SOH')
x_train3= train_data3[inputs]
x_test3= test_data3[inputs]
y_train3= train_data3[target]
y_test3= test_data3[target]
x_train3 = scaler.fit_transform(x_train3)
x_test3 = scaler.transform(x_test3)
x_train_array3 = x_train3.reshape((x_train3.shape[0], 1, x_train3.shape[1]))
x_test_array3 = x_test3.reshape((x_test3.shape[0], 1, x_test3.shape[1]))

#----------------------------------------------- RNN GRU ---------------------------------------------------------------------------------------------------------
modelGRU_100 = Sequential()
modelGRU_100.add(GRU(units=256, return_sequences=True, input_shape=(1, 7)))
modelGRU_100.add(Dense(units=1, activation='linear'))
modelGRU_100.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
modelGRU_100.summary()
history_GRU100 = modelGRU_100.fit(x_train_array3, y_train3, batch_size=64, epochs=100, validation_data=(x_test_array3, y_test3),callbacks=[callback])
plt.plot(history_GRU100.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_GRU100.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = modelGRU_100.evaluate(x_test_array3, y_test3, verbose=1)
print('Test Loss: {}'.format(results))

#----------------------------------------------- RNN LSTM ------------------------------------------------------------------------------------------------------

model_LSTM100 = Sequential()
model_LSTM100.add(LSTM(units=256, return_sequences=True, input_shape=(1, 7)))
model_LSTM100.add(Dropout(0.2))
model_LSTM100.add(Dense(units=1, activation='relu'))
model_LSTM100.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_LSTM100.summary()
history_LSTM100 = model_LSTM100.fit(x_train_array3, y_train3, batch_size=64, epochs=100, validation_data=(x_test_array3, y_test3),callbacks=[callback])
plt.plot(history_LSTM100.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_LSTM100.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_LSTM100.evaluate(x_test_array3, y_test3, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_lstm_100 = model_LSTM100.predict(x_test_array3)

#----------------------------------------------- ANN ----------------------------------------------------------------------------------------------------------

model_ANN100 = Sequential()
model_ANN100.add(Dense(units=256, activation='relu', input_shape=(1,7)))  
model_ANN100.add(Dense(units=256, activation='relu'))
model_ANN100.add(Dense(units=1, activation='linear'))
model_ANN100.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_ANN100.summary()
history_ANN100 = model_ANN100.fit(x_train_array3, y_train3, batch_size=64, epochs=100, validation_data=(x_test_array3, y_test3),callbacks=[callback])
plt.plot(history_ANN100.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_ANN100.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_ANN100.evaluate(x_test_array3, y_test3, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_ann_100 = model_ANN100.predict(x_test_array3)

#----------------------------------------------- CNN LSTM -----------------------------------------------------------------------------------------------------

model_CNNLSTM100 = Sequential()
model_CNNLSTM100.add(TimeDistributed(Conv1D(128, 3, activation='relu'),
                          input_shape=(1,7,1)))
model_CNNLSTM100.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model_CNNLSTM100.add(TimeDistributed(Flatten()))
model_CNNLSTM100.add(LSTM(256,return_sequences=True))
model_CNNLSTM100.add(LSTM(256,return_sequences=False))
model_CNNLSTM100.add(Dropout(0.2))
model_CNNLSTM100.add(Dense(units=1, activation='relu'))
model_CNNLSTM100.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model_CNNLSTM100.summary()
history_CNNLSTM100 = model_CNNLSTM100.fit(x_train_array3, y_train3, batch_size=64, epochs=100, validation_data=(x_test_array3, y_test3),callbacks=[callback])
plt.plot(history_CNNLSTM100.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_CNNLSTM100.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = model_CNNLSTM100.evaluate(x_test_array3, y_test3, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_cnn_lstm_100 = model_CNNLSTM100.predict(x_test_array3)

#----------------------------------------------- CNN LSTM ATT -------------------------------------------------------------------------------------------------

def cnn_lstm_att100_after():
    inputs = Input(shape=(1,7,1))
    out=TimeDistributed(Conv1D(128, 3, activation='relu'))(inputs)
    out=TimeDistributed(MaxPooling1D(pool_size=1))(out)
    out=TimeDistributed(Flatten())(out)
    out=Bidirectional(LSTM(256,return_sequences=True))(out)
    out=Bidirectional(LSTM(256,return_sequences=True))(out)
    attention_mul = attention_3d_block(out)
    output = Dense(1, activation='relu')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

cnn_lstm_att100=cnn_lstm_att100_after()
cnn_lstm_att100.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
print(cnn_lstm_att100.summary())
history_cnn_lstmatt100=cnn_lstm_att100.fit(x_train_array3, y_train3, epochs=100, batch_size=64, validation_data=(x_test_array3, y_test3), callbacks=[callback])
plt.plot(history_cnn_lstmatt100.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_cnn_lstmatt100.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('cnn_lstm_att MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
results = cnn_lstm_att100.evaluate(x_test_array3, y_test3, verbose=1)
print('Test Loss: {}'.format(results))
y_pred_cnn_lstm_att_100 = cnn_lstm_att100.predict(x_test_array3)

#--------------------------- PREDICCIONES Y ERRORES -------------------------------------------------------------------------------------------

dfs['B0005']['SOH_LSTM_100']= y_pred_lstm_100.reshape(-1)
dfs['B0005']['SOH_ANN_100']= y_pred_ann_100.reshape(-1)
dfs['B0005']['SOH_CNN-LSTM_100']= y_pred_cnn_lstm_100.reshape(-1)
dfs['B0005']['SOH_CNN-LSTM-ATT_100']=y_pred_cnn_lstm_att_100.reshape(-1)
for col in ['SOH_LSTM_100','SOH_ANN_100','SOH_CNN-LSTM_100','SOH_CNN-LSTM-ATT_100'] :
    dfs['B0005'][f'Error(%)_{col}'] = np.abs(dfs['B0005'][col] - dfs['B0005']['SOH']) / dfs['B0005']['SOH'] * 100

df_plot = dfs['B0005'].groupby('cycle').mean().reset_index()

# Representamos las predicciones 
plt.figure(figsize=(10, 6))
plt.plot(df_plot['cycle'], df_plot['SOH_LSTM_100'], label='LSTM Predicted SOH')
plt.plot(df_plot['cycle'], df_plot['SOH_ANN_100'], label='ANN Predicted SOH')
plt.plot(df_plot['cycle'], df_plot['SOH_CNN-LSTM_100'], label='CNN-LSTM Predicted SOH')
plt.plot(df_plot['cycle'], df_plot['SOH_CNN-LSTM-ATT_100'], label='CNN-LSTM-ATT Predicted SOH')
plt.plot(df_plot['cycle'], df_plot['SOH'], label='Test SOH',color='black', linewidth=2)
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH')
plt.title('Model Prediction Comparison (whole lifecycle train data)')
plt.legend()
plt.grid(True)
plt.show()

# Representamos los errores relativos de las predicciones 
plt.figure(figsize=(10, 6))
plt.plot(df_plot['cycle'], df_plot['Error(%)_SOH_LSTM_100'], label='LSTM Prediction Error')
plt.plot(df_plot['cycle'], df_plot['Error(%)_SOH_ANN_100'], label='ANN Prediction Error')
plt.plot(df_plot['cycle'], df_plot['Error(%)_SOH_CNN-LSTM_100'], label='CNN-LSTM Prediction Error')
plt.plot(df_plot['cycle'], df_plot['Error(%)_SOH_CNN-LSTM-ATT_100'], label='CNN-LSTM-ATT Prediction Error')
plt.xlabel('Discharge Cycle')
plt.ylabel('SOH Prediction Error (%)')
plt.title('Model Prediction Error Comparison (whole lifecycle train data)')
plt.legend()
plt.grid(True)
plt.show()

