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

class Submitfull(Modelado):

    def submit(self, data_type):
        df_transform = self.transform_df(df=self.df_final, type_transform="returns",remove_na=True)
        modelos_info_criteria = self.apply_model(df_transform, max_iters=21, data_type=data_type) # Remover criterio adicional
        test_results = self.apply_final_model(modelos_info_criteria, df_transform)
        test_results.to_csv(f"Resultados_todos_modelos_{data_type}_datos.csv",index=False)
        return test_results
