from Portafolio.Portafolio import Portafolio

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

class Modelado(Portafolio):
    
    def adfuller_test_results(self,b):
        # pruebas de estacionarieidad para todas las series
        results_adftest = {}
        for column in b.columns.tolist():
            print(column)
            adf = adfuller(b[column])
            results_adftest[column] = adf[1]

        vals = np.fromiter(results_adftest.values(), dtype=float)

        return pd.DataFrame(results_adftest.items(), columns=["Ticker","p-value"])
    
    def var_model_results(self,b,max_iters):
        models_results_by_lags = {}
        model = VAR(b)
        iters = np.arange(1,max_iters)
        for i in iters:
            result = model.fit(i)
            models_results_by_lags[i] = {"aic":result.aic,
                                "bic":result.bic,
                                'fpe': result.fpe,
                                'hqic':result.hqic}
        
        # list_AIC = self.iter_var("AIC",max_iters,models_results_by_lags)
        # list_BIC = self.iter_var("BIC",max_iters,models_results_by_lags)
        # list_FPE = self.iter_var("FPE",max_iters,models_results_by_lags)
        # list_HQIC = self.iter_var("HQIC",max_iters,models_results_by_lags)

        # return (list_AIC,list_BIC,list_FPE,list_HQIC)
        return models_results_by_lags

    def durbin_watson_df(self):
        out = durbin_watson(self.results.resid)
        df_out = pd.DataFrame(out, columns={"result"})
        df_out["no_correlation"] = df_out["result"].apply(lambda x: x>1.85 and x<2.15)
        return df_out

    # def grangers_causation_matrix(self, data):    
    #     salida = pd.DataFrame(Portafolio.ids_filters, columns = ["var1","var2"])
    #     salida["p-value"] = np.nan

    #     for id,pair in enumerate(Portafolio.ids_filters):
    #         test_result = grangercausalitytests(data[list(pair)], maxlag=[self.maxlag], verbose=False) # remover parentesis en maxlag
    #         # p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]     
    #         p_values = round(test_result[self.maxlag][0][self.test][1],4)
    #         # min_p_value = np.min(p_values)
    #         # salida.iloc[id,2] = min_p_value
    #         salida.iloc[id,2] = p_values
    #     return salida

    def iter_var(self,criterio,max_lag,models_results_by_lags):
        salida = [models_results_by_lags[id][criterio] for id in np.arange(1,max_lag)]
        max_lag = salida.index(min(salida))+1
        print(f"{criterio}: ",max_lag)
        return max_lag
    
    def clean_output_model(self,df):
        final_df = df.copy()
        final_df["pass_test"] = final_df["p-value"]<0.05
        final_df = final_df[final_df["pass_test"]==True]
        return final_df    
        
        
    def apply_model(self,df,max_iters,use_file_criteria=False,data_type="full"):
        if use_file_criteria:
            new_df = pd.read_csv("modelos_info_criteria.csv")
            if list(new_df.columns)[0] == "Unnamed: 0":
                new_df.drop(["Unnamed: 0"],axis=1,inplace=True)

        else:
            model = [self.var_model_results(df[list(column)],max_iters) for column in self.ids_filters]
            print("termine modelos")
            new_df = pd.DataFrame(self.ids_filters,columns=["var1","var2"])
            new_df["aic"] = [self.iter_var("aic",max_iters,model_iter) for model_iter in model]
            print("termine AIC")
            new_df["bic"] = [self.iter_var("bic",max_iters,model_iter) for model_iter in model]
            print("termine BIC")
            new_df["fpe"] = [self.iter_var("fpe",max_iters,model_iter) for model_iter in model]
            print("termine FPE")
            new_df["hqic"] = [self.iter_var("hqic",max_iters,model_iter) for model_iter in model]
            print("termine HQIC")

            # Escritura de archivo
            new_df.to_csv(f"{data_type}_modelos_info_criteria.csv",index=False)

        return new_df
    
    def modelito(self,df,df_transform):

        modelos_lista = {}
        for row in np.arange(0,df.shape[0]):
            s = df_transform[[df.loc[row,"var1"],df.loc[row,"var2"]]]
            model = VAR(s)
            modelos_lista[row] = model
        
        self.modelos_lista = modelos_lista
    
    def criteria_selection_codition(self,df):

        criterio_a = df['aic']
        criterio_b = df['bic']
        criterio_c = df["fpe"]

        if (df['aic'] == criterio_a) & (df['bic'] == criterio_a) & (df['fpe'] == criterio_a) & (df["hqic"] == criterio_a):
            return criterio_a,"aic"
        
        elif (df['aic'] == criterio_a) & (df['fpe'] == criterio_a):
            return criterio_a,"aic"
        
        elif (df['hqic'] == criterio_b) & (df['bic'] == criterio_b):
            return criterio_b,"bic"

        else:
            return criterio_c,"fpe"
    
    def apply_best_criteria(self,df):

        new_df = df.apply(self.criteria_selection_codition, axis='columns',result_type='expand')

        df = pd.concat([df,new_df],axis='columns')
        df.rename(columns = {0:"maxlags",1:"criteria"}, inplace=True)

        return df
    
    def save_fit_models(self,maxlags,criteria,index_row):
        
        model = self.modelos_lista[index_row]
        result = model.fit(maxlags=maxlags, ic= criteria)
        self.lista_results[index_row] = result
    
    def fit_models_over_best_criteria(self,result):

        norm = result.test_normality()
        pvalue_normality = norm.pvalue
        normality = pvalue_normality<0.05

        try:
            stability = result.is_stable()
        except:
            stability = "Error"

        return pvalue_normality, normality, stability
    
    def apply_vif(self,df):
        
        try:
            vif = [variance_inflation_factor(df,id) for id in np.arange(0,2)]
            collinearity_var1 = vif[0]
            collinearity_var2 = vif[1]
        
        except:
            collinearity_var1 = "Error"
            collinearity_var2 = "Error"

        return collinearity_var1,collinearity_var2
    
    def apply_vif_results(self,df,df_transform):
        
        results = []

        for row in np.arange(0,df.shape[0]):
            new_df = df_transform[[df.loc[row,"var1"],df.loc[row,"var2"]]]
            salida = self.apply_vif(new_df)
            results.append(salida)

        results = pd.DataFrame(results,columns=["collinearity_var1","collinearity_var2"])
        print(results.shape)
        print(results.isna().sum())
        df = pd.concat([df,results],axis='columns')
        return df

    def apply_test_causality(self):
        
        list_results = []
        for id in np.arange(0,len(self.lista_results)):
            try:
                model_result = self.lista_results[id]
                test = model_result.test_causality(causing=list(model_result.params.columns)[0],caused=list(model_result.params.columns)[1])
                p_value_causality = test.pvalue
                causality = p_value_causality<0.05
                list_results.append((p_value_causality,causality))
            except:
                p_value_causality = "Error"
                causality = "Error"
                list_results.append((p_value_causality,causality))
        
        results = pd.DataFrame(list_results,columns=["p_value_causality","causality"])

        return results

    def row_name(self,df):
        df["index_row"] = df.apply(lambda x: x.name, axis='columns')
        return df

    def apply_fit(self,df):
        
        new_df = []
        # new_df = df.apply(self.fit_models_over_best_criteria, axis = 'columns', result_type='expand')
        for row in np.arange(0,len(self.lista_results)):
            result = self.lista_results[row]
            salida = self.fit_models_over_best_criteria(result)
            new_df.append(salida)

        new_df = pd.DataFrame(new_df,columns=["pvalue_normality","normality","stability"])
        df = pd.concat([df,new_df], axis='columns')

        return df

    def apply_final_model(self,df,df_transform):

        self.modelito(df,df_transform)

        # Select best criteria results
        best_model_df = self.apply_best_criteria(df)

        # Select best criteria results
        test_results = self.row_name(best_model_df)

        # Apply test results (normality,stability)
        test_results.apply(lambda x: self.save_fit_models(maxlags= x.maxlags, criteria=x.criteria, index_row= x.index_row),axis='columns')
        test_results = self.apply_fit(best_model_df)
        
        # VIF results colineality test
        test_results = self.apply_vif_results(test_results,df_transform)

        # Causality test
        causality = self.apply_test_causality()
        test_results = pd.concat([test_results,causality], axis='columns')

        return test_results
                                