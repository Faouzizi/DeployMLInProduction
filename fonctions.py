###############################################################
##### import packages
###############################################################
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import reverse_geocode
from sklearn.preprocessing import StandardScaler

def imputation_statique(df, statique):
    ###############################################################
    # Cette fonction vous permettra d'imputer les données manquantes
    # Si statique=True alors l'imputation se fera par la median ou le mode
    # selon le type des données en entrée
    ###############################################################
    missing_data = df.apply(lambda x: np.round(x.isnull().value_counts()*100.0/len(x),2)).iloc[0]
    columns_MissingData = missing_data[missing_data<100].index
    if imputation_statique:
        for col in columns_MissingData:
            if df[col].dtype=='O':
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    else:
        imputer = KNNImputer(n_neighbors=3)
        ids = df.CustomerID
        X = pd.concat([pd.get_dummies(df.drop('CustomerID', axis=1).select_dtypes('O')), df.drop('CustomerID', axis=1).select_dtypes(exclude='O')], axis=1)
        X_filled_knn = pd.DataFrame(imputer.fit_transform(X))
        X_filled_knn.columns = X.columns
        for col in columns_MissingData:
            print(col)
            if df[col].dtypes=='O':
                df_temp =X_filled_knn.filter(regex='^'+col+'*')
                df_temp.columns = [x.replace(col+'_', '') for x in df_temp.columns]
                df[col] = df_temp.idxmax(1)
            else:
                df[col] = np.round(X_filled_knn[col],2)
    return(df)

def traiter_valeurs_extremes_continues_prod(df,df_origin, variable_cible):
    ###############################################################
    # Cette fonction vous permettra de traiter les valeurs extrèmes
    # les valeurs extrèmes seront remplacé par moyenne dans ce cas
    ###############################################################
    for col in df_origin.select_dtypes(exclude='O'):
        if col != variable_cible:
            q1 = df_origin[col].quantile([0.25]).values[0]
            q3 = df_origin[col].quantile([0.75]).values[0]
            IC_valeur_non_aberantes = [q1 - 2*(q3-q1), q3 + 2*(q3-q1)]
            df.loc[df[col]<IC_valeur_non_aberantes[0], col] = df_origin[col].mean()
            df.loc[df[col]>IC_valeur_non_aberantes[1], col] = df_origin[col].mean()
    return(df)


def get_newData_processed(df, val=False):
    #########################################################################################################
    ##############     On vérifie que toutes les données sont au bon format       ###########################
    #########################################################################################################
    df['Income'] = pd.to_numeric(df['Income'],errors='coerce')
    # on remplace la variable de geolocalisation par la variable country
    coordonnees = tuple(df['Location.Geo'].map(lambda x: tuple(x.split(',')) if x.split(',')[0]!='NA' else (17,77)))
    df['country'] = [x['country'] for x in reverse_geocode.search(coordonnees)]
    # on supprime la varibale de geolocalisation
    df.drop('Location.Geo', axis=1, inplace=True)
    #/////// si il y a des données manquantes faire :
    df_origine = pd.read_csv('/Volumes/Carte_mem/medium/Customer_LifeTime_Value/customerLifetimeValue/data methode 2 withoutMissingData ExtremesValues.csv')
    try:
        df_origine = df_origine.drop('Unnamed: 0', axis=1)
    except:
        pass
    if df.apply(lambda x: x.isnull().value_counts()).shape[0]>1:
        try:
            df_origine = df_origine.drop('Customer.Lifetime.Value', axis=1)
        except:
            pass
        #########################################################################################################
        ##############               On imputeles valeurs manquantes                  ###########################
        #########################################################################################################
        df_imputation = pd.concat([df_origine, df], axis=0)
        imputation_statique(df_imputation, statique='')
        df = df_imputation[df_imputation.CustomerID.isin(df.CustomerID.unique())]
    ########################################################################################################
    ##############           TRAITEMENT VALEURS EXTREMES        ###########################
    #########################################################################################################
    df = traiter_valeurs_extremes_continues_prod(df,df_origine, variable_cible='Customer.Lifetime.Value')
    #########################################################################################################
    ##############                                    Data processing                  ###########################
    #########################################################################################################
    df = pd.concat([df, df_origine.drop('Customer.Lifetime.Value', axis=1)], axis=0)
    variables_explicatives = df.drop(['CustomerID'], axis=1)
    variables_explicatives['Monthly.Premium.AutoSquare'] = variables_explicatives['Monthly.Premium.Auto']**2
    variables_explicatives['Total.Claim.AmountSquare'] = variables_explicatives['Total.Claim.Amount']**2
    variables_explicatives['Number.of.Open.ComplaintsSquare'] = variables_explicatives['Number.of.Open.Complaints']**2
    variables_explicatives['IncomeSquare'] = variables_explicatives['Income']**2
    variables_explicatives['Months.Since.Last.ClaimSquare'] = variables_explicatives['Months.Since.Last.Claim']**2
    variables_explicatives['Months.Since.Policy.InceptionSquare'] = variables_explicatives['Months.Since.Policy.Inception']**2
    variables_explicatives['Months.Since.Policy.InceptionSquare'] = variables_explicatives['Months.Since.Policy.Inception']**2
    variables_explicatives['Number.of.PoliciesSquare'] = variables_explicatives['Number.of.Policies']**2
    variables_explicatives_continues = variables_explicatives.select_dtypes('float')
    ### One hot encoding
    variables_explicatives_qualitatives = variables_explicatives.select_dtypes('O')
    variables_explicatives_qualitatives = pd.get_dummies(variables_explicatives_qualitatives)
    variables_explicatives = pd.concat([variables_explicatives_continues, variables_explicatives_qualitatives], axis=1)
    ### redimensionnement
    scaler = StandardScaler()
    scaler.fit(variables_explicatives)
    colonnes = variables_explicatives.columns
    variables_explicatives = pd.DataFrame(scaler.transform(variables_explicatives))
    variables_explicatives = pd.DataFrame(variables_explicatives.iloc[0]).T
    variables_explicatives.columns = colonnes
    return(variables_explicatives)

#df = pd.read_csv('/Volumes/Carte_mem/medium/Customer LifeTime Value/customerLifetimeValue/data methode 2.csv')
#df.drop('Customer.Lifetime.Value', axis=1, inplace=True)
#df.CustomerID = df.CustomerID+99999
#df = get_newData_processed(df, val=False)
