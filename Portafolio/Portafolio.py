import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools


from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.outliers_influence import variance_inflation_factor

# estacionarieidad
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# test durbin watson
from statsmodels.stats.stattools import durbin_watson

class Portafolio:

    INDICES = ["76_CHL_IGPA_INDICE","77_COL_COLCAP_INDICE","78_MEX_IPC_INDICE","79_PER_LIMA GENER_INDICE"]
    PERIODO = ["PERIODO"]

    def __init__(self,df):
        self.df = df
        self.tickers_info_full = self.df.columns.to_list()
        self.tickers_info_full.remove("PERIODO")
        [self.tickers_info_full.remove(indice) for indice in Portafolio.INDICES]
        self.df_final = self.df[self.tickers_info_full]
        self.test = 'ssr_chi2test'
        self.inside = False
        self.lista_results = {}
    
    def create_df_tickers(self,list_tickers,inside=False):
        tickers_info_df = pd.DataFrame(list_tickers, columns = ["column_id"])
        if inside:
            value_pais=2
        else:
            value_pais=1
        tickers_info_df["pais"] = tickers_info_df["column_id"].apply(lambda x: x.split("_")[value_pais])
        self.tickers_info_df = tickers_info_df

    def permutations_tickers(self, *args, **kwargs):
        s = list(itertools.product(*args, **kwargs))
        ids = dict()

        if self.inside:
            value_pais = 2
        else:
            value_pais = 1

        for id in s:
            value_1 = id[0].split("_")[value_pais]
            value_2 = id[1].split("_")[value_pais]
            if value_1==value_2:
                ids[id] = 1 # mismo pais
            else:
                ids[id] = 0

        ids_filters = [id for id in ids if ids[id] == 0]
        self.ids_filters = ids_filters
        
    def analysis_df(self,par_var):
        lista = Portafolio.PERIODO.copy()
        lista.extend(par_var)
        new_df = self.df.copy()
        new_df = new_df[lista]
        new_df = new_df.set_index("PERIODO")
        return new_df

    def transform_df(self,df,type_transform=None,remove_na=None):
        
        df_edit = df.copy()
        new_columns_org = df_edit.columns.tolist()

        if type_transform=="returns":            
            new_columns = ["returns_"+column for column in new_columns_org]

            for column in new_columns:
                df_edit[column] = np.log(df_edit[column.replace("returns_","")]).diff()
        
        elif type_transform=="diff":
            new_columns = ["diff_"+column for column in new_columns_org]

            for column in new_columns:
                df_edit[column] = df_edit[column.replace("diff_","")].diff()

        if type_transform:
            df_edit.drop(new_columns_org,axis=1,inplace=True)
        
        # calidad de la salida
        print("Nas:",df_edit.isna().sum())

        if remove_na:
            df_edit.dropna(inplace=True,axis=0)
            print("Correccion")
            print("Nas:",df_edit.isna().sum())

        self.inside = True
        self.apply_all(inside=True,df=df_edit.columns.tolist())
        
        return df_edit
         
    def apply_all(self,inside=False,df=None):

        if inside:
            self.create_df_tickers(list_tickers=df,inside=True)
        
        else:
            self.create_df_tickers(list_tickers=self.tickers_info_full)
        
        self.permutations_tickers(self.tickers_info_df["column_id"],repeat=2)        