from Portafolio.Portafolio import Portafolio
from Portafolio.Modelado import Modelado

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.outliers_influence import variance_inflation_factor

# estacionarieidad
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# test durbin watson
from statsmodels.stats.stattools import durbin_watson

# escritura modelos
import pickle 

class Submitfull(Modelado):

    def __init__(self, df, data_type):
        self.data_type = data_type
        super().__init__(df)
        
    def submit(self):

        # Aplicación de transformaciones
        df_transform = self.transform_df(df=self.df_final, type_transform="returns",remove_na=True)
        self.df_transform = df_transform

        # Aplicación de modelos
        modelos_info_criteria = self.apply_model(self.df_transform, max_iters=21, data_type=self.data_type)
        self.modelos_info_criteria = modelos_info_criteria

        # Aplicación de test necesarios
        test_results = self.apply_final_model(self.modelos_info_criteria, self.df_transform)
        test_results.to_csv(f"Resultados_todos_modelos_{self.data_type}_datos.csv",index=False)

        self.test_results = test_results

        self.filter_test_results = self.filter_data(self.test_results)
        self.filter_test_results.to_csv(f"Resultados_finales_datos_{self.data_type}.csv",index=False)

        # Archivo de tickers definitivo
        id_modelos = pd.DataFrame(self.ids_filters,columns= ["var1","var2"])
        id_modelos.to_csv("../Datos/Ids_modelos.csv",index=False)

        return test_results

    def escritura_info(self,type="modelo"):
        
        if type=="modelo":
            ruta = "Modelos"
            salida = self.modelos_lista

        elif type=="resultados":
            ruta = "Resultados"
            salida = self.lista_results
        
        for modelo in np.arange(0,len(self.modelos_lista)):
            
            if type=="modelo":
                path = f'../{ruta}/{self.data_type}_model_{modelo}.pkl'
            
            else:
                path = f'../{ruta}/{self.data_type}_result_{modelo}.pkl'

            with open(path,"wb") as f:
                modelo = salida[modelo]                    
                pickle.dump(modelo,f)
    
    def lectura_info(self, type="modelo"):

        if type=="modelo":
            ruta = "Modelos"

        elif type=="resultados":
            ruta = "Resultados"
        
        ids_modelos = pd.read_csv("../Datos/Ids_modelos.csv")

        modelos_full = {}
        for modelo in np.arange(0,ids_modelos.shape[0]):

            if type=="modelo":
                path = f'../{ruta}/{self.data_type}_model_{modelo}.pkl'
            
            else:
                path = f'../{ruta}/{self.data_type}_result_{modelo}.pkl'

            with open(path,'rb') as f:
                modelos_full[modelo] = pickle.load(f)
        
        return modelos_full