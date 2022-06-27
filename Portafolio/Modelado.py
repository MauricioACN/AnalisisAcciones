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

# test het_white
from statsmodels.stats.diagnostic import het_white 
from statsmodels.compat import lzip
from patsy import dmatrices

class Modelado(Portafolio):
    
    def adfuller_test_results(self,b):
        # pruebas de estacionarieidad para todas las series
        results_adftest = {}
        for column in b.columns.tolist():
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
        
        return models_results_by_lags

    def durbin_watson_test(self, results):
        out = durbin_watson(results)
        var_1 = out[0]
        var_2 = out[1]

        no_correlation_var1 = (var_1 > 1.85) & (var_1 < 2.15)
        no_correlation_var2 = (var_2 > 1.85) & (var_2 < 2.15)

        return (var_1, var_2, no_correlation_var1, no_correlation_var2)

    def apply_durbin_watson(self):
    
        list_results = []
        for id in np.arange(0,len(self.lista_results)):
            try:
                model_result = self.lista_results[id]
                test = self.durbin_watson_test(model_result.resid)
            except:
                test = ("Error",)*4
            
            list_results.append(test)

        results = pd.DataFrame(list_results,columns=["p_value_no_correlation_var_1", "p_value_no_correlation_var_2", "no_correlation_var1", "no_correlation_var2"])

        return results

    def cointegration_test(self,y0,y1,maxlags = 1, trend="c"):

        out = coint(y0, y1, maxlag = maxlags, trend = trend)
        pvalue_coint = out[1] # valor p psotion 
        t_stadistic = out[0] # t statistic
        criteria_info_5 = out[2][1]
        cointegration = t_stadistic<criteria_info_5

        return pvalue_coint,t_stadistic,criteria_info_5,cointegration

    def cointegration_johansen_test(self, var1, var2, k_ar_diff):
        
        det_order = 0
        df = self.df_transform[[var1,var2]]
        test = coint_johansen(df,det_order,k_ar_diff)

        eigenvalue = test.lr2[0]
        critical_values_eigen_95 = test.cvm[0][1]
        cointegration_joha = eigenvalue > critical_values_eigen_95

        return eigenvalue, critical_values_eigen_95, cointegration_joha

    def apply_cointegrations_test(self,var1,var2,maxlags):
        
        y0 = self.df_transform[var1]
        y1 = self.df_transform[var2]

        try:
            test = self.cointegration_test(y0,y1,maxlags = maxlags)
            
        except:
            test = ("Error",)*2
        
        return test

    def iter_var(self,criterio,max_lag,models_results_by_lags):
        salida = [models_results_by_lags[id][criterio] for id in np.arange(1,max_lag)]
        max_lag = salida.index(min(salida))+1
        return max_lag
    
    def filter_data(self,df):

        df_final = df.copy() 
        for column in self.PASS_TEST:
            if column not in ["collinearity_var2","collinearity_var1"]:
                df_final = df_final[(df_final[column]=="True") | (df_final[column]==True)]
            else:
                df_final = df_final[df_final[column]<5]
        return df_final
        
    def apply_model(self,df,max_iters,use_file_criteria=False,data_type="full"):
        if use_file_criteria:
            new_df = pd.read_csv("modelos_info_criteria.csv")
            if list(new_df.columns)[0] == "Unnamed: 0":
                new_df.drop(["Unnamed: 0"],axis=1,inplace=True)

        else:
            model = [self.var_model_results(df[list(column)],max_iters) for column in self.ids_filters]
            new_df = pd.DataFrame(self.ids_filters,columns=["var1","var2"])
            new_df["aic"] = [self.iter_var("aic",max_iters,model_iter) for model_iter in model]
            new_df["bic"] = [self.iter_var("bic",max_iters,model_iter) for model_iter in model]
            new_df["fpe"] = [self.iter_var("fpe",max_iters,model_iter) for model_iter in model]
            new_df["hqic"] = [self.iter_var("hqic",max_iters,model_iter) for model_iter in model]

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
    
    def het_white_test(self, index):

        a = self.lista_results[index]
        mode = self.modelos_lista[index]

        print(mode.endog_names)
        print(self.df_transform.columns)
        endog = self.df_transform[mode.endog_names]
        endog = endog.filter(items = a.resid.index, axis=0)
        endog.columns = ["var1","var2"]

        expr = 'var1~ var2'
        y, X = dmatrices(expr, endog, return_type='dataframe')
        try:
            results = het_white(a.resid.iloc[:,0], X)
        except:
            results = ("Error",)*4

        return results


    def apply_final_model(self,df,df_transform, lag_model=False):

        self.modelito(df,df_transform)

        # Select best criteria results
        best_model_df = self.apply_best_criteria(df)

        if lag_model:
            filtro_colums = pd.read_excel("../Datos/SubFiltro.xlsx",sheet_name=0)

            if self.type_transform=="returns":
                filtro_colums["var1"] = "returns_"+filtro_colums["var1"]
                filtro_colums["var2"] = "returns_"+filtro_colums["var2"]

            elif self.type_transform=="diff":
                filtro_colums["var1"] = "diff_"+filtro_colums["var1"]
                filtro_colums["var2"] = "diff_"+filtro_colums["var2"]

            best_model_df = best_model_df.merge(filtro_colums, on = ["var1","var2"], how = "inner")

            best_model_df["maxlags_2"] = np.where(best_model_df["maxlags"]==best_model_df["Lags"], best_model_df["maxlags"],
                                                np.where(best_model_df["Lags"].isna(), best_model_df["maxlags"], best_model_df["Lags"]))
            
            best_model_df.drop(["maxlags","Lags","corr"], axis = 1, inplace = True)
            best_model_df.rename(columns={"maxlags_2":"maxlags"}, inplace = True)
        
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

        # Durbin Watson
        dwatson_test = self.apply_durbin_watson()
        test_results = pd.concat([test_results,dwatson_test], axis='columns')

        # # Cointegration
        # Cointegration = test_results.apply(lambda x: self.apply_cointegrations_test(x.var1, x.var2, x.maxlags), axis = 'columns')
        # Cointegration = pd.DataFrame(list(Cointegration), columns = ["pvalue_coint","t_stadistic","criteria_info_5","cointegration"])
        
        # test_results = pd.concat([test_results,Cointegration],axis="columns")


        # heterocedasticity
        het_wthite_df = test_results.apply(lambda x: self.het_white_test(x.index_row), axis = 'columns')
        het_wthite_df = pd.DataFrame(list(het_wthite_df), columns = ['H_LM_statistic', 'H_LM_test_p_value:', 'H_F_statistic', 'H_F-test_p_value'])
        test_results = pd.concat([test_results,het_wthite_df],axis="columns")


        # Cointegration Johansen
        johansen = test_results.apply(lambda x: self.cointegration_johansen_test(var1 = x.var1, var2 = x.var2, k_ar_diff = x.maxlags), axis = 'columns')
        johansen_df = pd.DataFrame(list(johansen), columns = ["eigenvalue", "critical_values_eigen_95", "cointegration_joha"])
        test_results = pd.concat([test_results,johansen_df],axis="columns")

        return test_results
                              