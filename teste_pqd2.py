import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import itertools




warnings.filterwarnings("ignore")
os.system('cls')

nome_arquivo = "3 - fipezap-serieshistoricas-20092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_fipezap = pd.read_excel(nome_arquivo, sheet_name='Preços_FipeZAP')
#df_fipezap['Data'] = pd.to_datetime(df_fipezap['Data'])
df_fipezap.index=pd.to_datetime(df_fipezap['Data'])


# Teste ADF para estacionaridade
resultado_adf = adfuller(df_fipezap['Preço Médio Venda'])
print("Estatística ADF:", resultado_adf[0])
print("p-valor:", resultado_adf[1])

if resultado_adf[1] < 0.05:
    print("A série é estacionária (d = 0)")
else:
    print("A série não é estacionária, diferenciando... (d = 1)")
    df_fipezap['coluna_valor_diferenciada'] = df_fipezap['Preço Médio Venda'].diff().dropna()

    # Teste ADF após a diferenciação
    resultado_adf_diferenciada = adfuller(df_fipezap['coluna_valor_diferenciada'].dropna())
    print("Estatística ADF após diferenciação:", resultado_adf_diferenciada[0])
    print("p-valor após diferenciação:", resultado_adf_diferenciada[1])

# Plotar PACF para encontrar p
plot_pacf(df_fipezap['Preço Médio Venda'].dropna(), lags=20)
plt.title('PACF - Definir p')
plt.show()

# Plotar ACF para encontrar q
plot_acf(df_fipezap['Preço Médio Venda'].dropna(), lags=20)
plt.title('ACF - Definir q')
plt.show()

""""
Processo geral para identificar pp, dd, qq:

    dd (diferenciação):
        Use o ADF Test para verificar se a série é estacionária.
        Se a série não for estacionária, aplique a diferenciação d=1d=1 e repita o ADF Test. Se necessário, aplique d=2d=2.

    pp (auto-regressivo):
        Após a série estar estacionária, observe o gráfico de PACF para identificar o número de defasagens com picos significativos. Isso te dá o valor de pp.

    qq (média móvel):
        Observe o gráfico de ACF para identificar o número de defasagens com picos significativos. Isso te dá o valor de qq.


"""