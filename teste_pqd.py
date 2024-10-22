
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import itertools

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf




warnings.filterwarnings("ignore")
os.system('cls')

nome_arquivo = "3 - fipezap-serieshistoricas-20092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_fipezap = pd.read_excel(nome_arquivo, sheet_name='Preços_FipeZAP')
#df_fipezap['Data'] = pd.to_datetime(df_fipezap['Data'])
df_fipezap.index=pd.to_datetime(df_fipezap['Data'])

#####################################################################################################

# Exemplo de plotagem do PACF
plot_pacf(df_fipezap['Preço Médio Venda'], lags=20)
plt.title('Função de Autocorrelação Parcial (PACF)')
plt.show()


# Teste ADF
resultado_adf = adfuller(df_fipezap['Preço Médio Venda'])

print("Estatística ADF:", resultado_adf[0])
print("p-valor:", resultado_adf[1])
# Interpretação
if resultado_adf[1] < 0.05:
    print("A série é estacionária (não precisa de diferenciação)")
else:
    print("A série não é estacionária (precisa de diferenciação)")


# Exemplo de plotagem do ACF
plot_acf(df_fipezap['Preço Médio Venda'], lags=20)
plt.title('Função de Autocorrelação (ACF)')
plt.show()