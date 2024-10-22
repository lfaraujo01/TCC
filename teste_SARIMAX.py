import os
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import auto_arima

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import itertools

def extrair_data_final(periodo):
    # Separar os meses e o ano
    meses, ano = periodo[:-5], periodo[-4:]
    meses = meses.split('-')  # Exemplo: ['fev', 'mar', 'abr']
    
    # Pegar o último mês
    mes_final = meses[-1]  # Último mês do intervalo

    # Dicionário para converter nomes de meses para números
    meses_dict = {
        'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
        'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
    }
    
    # Converter o mês final para número
    mes_final_num = meses_dict[mes_final]
    
    # Criar a data com o primeiro dia do mês final (YYYY-MM-DD)
    data_final = pd.to_datetime(f'{ano}-{mes_final_num}-01')
    
    return data_final

def modelo_intervencao(df1, coluna_data, coluna_valor, data_intervencao, nome_arquivo, title, eixo_y, S=12):
    
    # Converter a coluna de datas para o formato datetime, se necessário
    df = df1.iloc[1:].copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    
    # Se a coluna de data já for o índice, evitar duplicidade
    if df.index.name != coluna_data:
        df = df.set_index(coluna_data)
    
    # Ordenar o DataFrame pela data
    df = df.sort_index()  # Usar o índice já ajustado (que é a data)
    
    # Criar uma variável indicadora de intervenção (0 antes, 1 a partir da data de intervenção)
    df['intervencao'] = (df.index >= pd.to_datetime(data_intervencao)).astype(int)
    
    # Verificar se a série é estacionária
    #estacionaria, adf_result, kpss_result = verificar_estacionaridade(df, coluna_valor)
    
    # Exibir resultados dos testes de estacionaridade
    #print("Resultados do teste ADF:")
    #print(f'Estatística ADF: {adf_result[0]}, Valor p: {adf_result[1]}')
    
    #print("\nResultados do teste KPSS:")
    #print(f'Estatística KPSS: {kpss_result[0]}, Valor p: {kpss_result[1]}')

    # Usar auto_arima para estimar os melhores parâmetros (p, d, q) e sazonais (P, D, Q, S)
    modelo_auto = pm.auto_arima(df[coluna_valor], 
                                seasonal=True, 
                                m=S,  # Sazonalidade (mensal, S=12)
                                exogenous=df[['intervencao']], 
                                start_p=0, start_q=0, max_p=3, max_q=3, # Limites para p e q
                                d=None,  # auto_arima determinará d automaticamente
                                D=None,  # auto_arima determinará D automaticamente
                                start_P=0, start_Q=0, max_P=3, max_Q=3, # Limites para P e Q
                                trace=True, # Exibir progresso
                                error_action='ignore', 
                                suppress_warnings=True)
    
    # Exibir o resumo do melhor modelo encontrado pelo auto_arima
    print("\nResumo do auto_arima:")
    print(modelo_auto.summary())
    
    # Extrair os parâmetros encontrados pelo auto_arima
    p, d, q = modelo_auto.order
    P, D, Q, S = modelo_auto.seasonal_order

    # Ajustar o modelo SARIMAX com os parâmetros encontrados
    modelo_sarimax = sm.tsa.SARIMAX(df[coluna_valor], 
                                    order=(p, d, q), 
                                    seasonal_order=(P, D, Q, S), 
                                    exog=df[['intervencao']])
    
    # Ajustar o modelo SARIMAX
    resultado = modelo_sarimax.fit()
    
    # Imprimir resumo do modelo ajustado
    print("\nResumo do SARIMAX ajustado:")
    print(resultado.summary())
    
    # Fazer previsões
    previsao = resultado.predict(start=2, end=len(df)-1, exog=df[['intervencao']].iloc[2:])
    
    # Plotar os resultados
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[coluna_valor], label='Série Original')
    plt.plot(df.index[2:], previsao, label='Previsão SARIMAX com Intervenção', color='red')
    plt.axvline(pd.to_datetime(data_intervencao), color='green', linestyle='--', label='Data de Intervenção')
    plt.legend()
    plt.ylabel(eixo_y)
    title += '- Modelo SARIMAX de Intervenção'
    plt.title(title)
    plt.show()

    # Extrair o resultado e salvar em Excel
    summary = resultado.summary()
    
    # Criar um DataFrame para as previsões
    df_previsao = pd.DataFrame({
        'Data': df.index,
        'Valor Real': df[coluna_valor],
        'Previsão': previsao
    }).set_index('Data')
    
    # Extrair os coeficientes do modelo (tabela principal)
    coef_table = pd.DataFrame(summary.tables[1].data[1:], columns=summary.tables[1].data[0])
    
    # Extrair informações gerais do modelo (tabela 0)
    info_table = pd.DataFrame(summary.tables[0].data)
    
    # Salvar os resultados em um arquivo Excel
    with pd.ExcelWriter(nome_arquivo) as writer:
        coef_table.to_excel(writer, sheet_name='Coeficientes')
        info_table.to_excel(writer, sheet_name='Informações_Gerais')
        df_previsao.to_excel(writer, sheet_name='Previsoes')
    
    return resultado


inicio_pandemia = '2020-03-06'

nome_arquivo = "4 - PNAD-20092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_pnad = pd.read_excel(nome_arquivo, sheet_name='PNAD')


# Aplicando a função
df_pnad['Data'] = df_pnad['Periodo'].apply(extrair_data_final)
df_pnad=df_pnad[['Data','Renda Média','Taxa de Desocupação','Periodo']]

title='Renda Média da População'
eixo_y='R$'
nome_arquivo='10-SARIMA_pnad_renda.xlsx'

resultado = modelo_intervencao(df_pnad, 'Data', 'Renda Média',inicio_pandemia,nome_arquivo,title,eixo_y)