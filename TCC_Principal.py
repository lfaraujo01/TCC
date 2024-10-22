###############################################################################
##  Trabalho de Conclusão de Curso - MBA Data Science e Analytics - DSA231  ##
##  Aluno: Lucas Fernandes Araujo - lfaraujo.eng@gmail.com                  ##
##  Orientador: Gustavo Lobo Dantas                                         ##
##  Data: 01/Outubro/2024                                                   ##
##############################################################################

# IMPORTAR BIBLIOTECAS ######################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

# Funções Criadas ###########################################################################

## Grava dataframe em arquivo excel. 
def grava_arquivo_excel(df, nome):
    # Garantir que o índice (Data) seja uma coluna antes de gravar o arquivo
    df_resetado = df.reset_index()

    # Gravar o DataFrame em um arquivo Excel
    saida = nome + ".xlsx"
    df_resetado.to_excel(saida, index=False)
    print(saida + " Gravado com Sucesso")

#Salva vários dataframes em um excel, cada um em uma aba
def salva_dfs_em_excel(dataframes, nome_arquivo):

    with pd.ExcelWriter(f"{nome_arquivo}.xlsx") as writer:
        for nome_df, df in dataframes.items():
            # Se a coluna Data não for o índice, resete o índice e mantenha a coluna Data
            if df.index.name != 'Data':
                df.reset_index(inplace=True)
            df.to_excel(writer, sheet_name=nome_df, index=False)
            print(f"DataFrame '{nome_df}' salvo na aba '{nome_df}' com sucesso.")

#Separa as séries do banco de dados SBPE.
def separa_base_SBPE(nome_series, nome_arquivo):
    # Leitura do arquivo CSV
    df = pd.read_csv(str(nome_arquivo))

    # Reordenar colunas para mover 'Info' para a primeira posição
    colunas = ['Info'] + [col for col in df.columns if col != 'Info']
    df = df[colunas]
    
    # Se nome_series for uma string única, converta para lista
    if isinstance(nome_series, str):
        nome_series = [nome_series]

    # Seleção das séries (uso de isin para filtrar múltiplas séries)
    df_filtrado = df[df['Info'].isin(nome_series)]

    # Ordenar pelo campo 'Data' em ordem crescente
    df_ordenado = df_filtrado.sort_values(by='Data')

    # Substituir vírgula por ponto na coluna 'Valor' e limpar pontos para evitar conflitos
    df_ordenado['Valor'] = df_ordenado['Valor'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)

    # Converter a coluna 'Valor' para tipo float
    df_ordenado['Valor'] = pd.to_numeric(df_ordenado['Valor'], errors='coerce')

    # Pivotar o DataFrame para que 'Data' seja o índice e 'Info' se torne as colunas
    df_pivotado = df_ordenado.pivot(index='Data', columns='Info', values='Valor')

    # Resetar o índice para 'Data' se tornar uma coluna normal (opcional)
    df_pivotado.reset_index(inplace=True)

    # Garantir que as colunas sejam na ordem correta e adicionar uma coluna 'Data'
    colunas_ordenadas = ['Data'] + [serie for serie in nome_series if serie in df_pivotado.columns]
    df_pivotado = df_pivotado[colunas_ordenadas]
    
    df_pivotado['Data'] = pd.to_datetime(df_pivotado['Data'])
    df_pivotado.set_index('Data', inplace=True)
    # Retornar o DataFrame pivotado
    
    return df_pivotado

#Plotagem de Série com marcação início e fim da pandemia
def plot_series(df, x_label, y_label, title,datas_marcadas):
    # Plotando a série temporal
    df.plot(title=title)

    # Marcando as datas específicas, se fornecidas
    if datas_marcadas:
        for date in datas_marcadas:
            plt.axvline(pd.to_datetime(date), color='red', linestyle='--', lw=2, label=f'Marca {date}')

    # Configurações adicionais do gráfico
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    
    
    # Remover notação científica no eixo Y
    plt.ticklabel_format(style='plain', axis='y')
    
    

    # Exibindo o gráfico
    plt.show()

def extrair_data_central(periodo):
    meses, ano = periodo[:-5], periodo[-4:]
    meses = meses.split('-')
    mes_central = meses[len(meses) // 2]  # Seleciona o mês do meio (central)
    
    # Dicionário para converter nomes de meses para números
    meses_dict = {
        'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
        'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
    }
    
    mes_central_num = meses_dict[mes_central]
    
    # Criando a data como o primeiro dia do mês central
    data_central = pd.to_datetime(f'{ano}-{mes_central_num}-01')
    
    return data_central

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

#Decomposição de serie temporal em tendencia, sazionalidade e resíduos.
def decompoe_serie_unica(df1, coluna_data, coluna_valor, title, datas_marcadas, eixo_y):
    
    df = df1.copy()
    
    # Definir a coluna 'Data' como índice
    df.set_index(coluna_data, inplace=True)
    
    # Criar uma série temporal com os valores
    series = pd.DataFrame(df[coluna_valor])
    
    # Plotar a série temporal
    series.plot(title=title)
    
    # Marcar datas específicas no gráfico
    for data in datas_marcadas:
        plt.axvline(pd.to_datetime(data), color='red', linestyle='--', lw=2, label=f'Marca {data}')
    
    plt.xlabel("Data")
    plt.ylabel(eixo_y)
    plt.grid(True)
    plt.show()
    
    # Decompor a série temporal
    serie_temporal = df[coluna_valor]
    #decomposição método aditivo (X = T + S + R)
    decomposicao = seasonal_decompose(serie_temporal, model='additive', period=12)
    
        ##### Teste de outras formas de decomposição para comparação de resultados ###############################
    
    #decomposição método multiplicativo (X = T x S x R)
    #decomposicao = seasonal_decompose(serie_temporal, model='multiplicative', period=12)
    
    #decomposição STL (método aditivo com logaritmo)
    #decomposicao = STL(serie_temporal, seasonal=12)
    
    ###########################################################################################################
    
    
    # Criar subplots para os componentes
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    title += '- Série Observada'
    # Plotar os componentes
    decomposicao.observed.plot(ax=ax[0], title=title)
    decomposicao.trend.plot(ax=ax[1], title='Tendência')
    decomposicao.seasonal.plot(ax=ax[2], title='Sazonalidade')
    decomposicao.resid.plot(ax=ax[3], title='Resíduos')
    
    # Marcar as datas específicas nos subplots
    for data in datas_marcadas:
        data_dt = pd.to_datetime(data)
        for axis in ax:
            axis.axvline(data_dt, color='red', linestyle='--', lw=2, label=f'Marca {data}')
    
    # Ajustes finais nos eixos
    for axis in ax:
        axis.grid(True)
    
    plt.tight_layout()
    plt.show()

def prever_serie_temporal(serie, ordem, passos_futuros):
    if not isinstance(serie.index, pd.DatetimeIndex):
        serie.index = pd.to_datetime(serie.index)
    
    serie = serie.dropna()
    
    if len(serie) == 0:
        raise ValueError("A série temporal está vazia após a remoção de nulos.")
    
    modelo = ARIMA(serie, order=ordem)
    modelo_ajustado = modelo.fit()

    previsao = modelo_ajustado.forecast(steps=passos_futuros)
    
    index_previsao = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), 
                                     periods=passos_futuros, 
                                     freq='ME')
    
    return pd.DataFrame(previsao, index=index_previsao, columns=['Previsão'])

def plotar_previsao(serie_original, previsao, titulo):
    plt.figure(figsize=(12, 6))
    plt.plot(serie_original, label='Série Original', color='blue')
    plt.plot(previsao, label='Previsão', color='orange', linestyle='--')
    plt.title(titulo)
    plt.xlabel('Data')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid()
    plt.show()

def verificar_estacionaridade(df, coluna_valor):
    y = df[coluna_valor]

    # Teste de Dickey-Fuller Aumentado (ADF)
    adf_result = adfuller(y)
    
    # Teste de Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
    kpss_result = kpss(y, regression='c')  # 'c' para constante

    # Verificar se a série é estacionária
    estacionaria = adf_result[1] <= 0.05 and kpss_result[1] > 0.05
    
    
    return estacionaria, adf_result, kpss_result

def modelo_intervencao(df1, coluna_data, coluna_valor, data_intervencao, nome_arquivo, title, eixo_y):
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
    estacionaria, adf_result, kpss_result = verificar_estacionaridade(df, coluna_valor)
    
    # Exibir resultados dos testes de estacionaridade
    print("Resultados do teste ADF:")
    print(f'Estatística ADF: {adf_result[0]}, Valor p: {adf_result[1]}')
    
    print("\nResultados do teste KPSS:")
    print(f'Estatística KPSS: {kpss_result[0]}, Valor p: {kpss_result[1]}')

    # Escolher o modelo ARIMA com base na estacionaridade
    if estacionaria:
        modelo = sm.tsa.ARIMA(df[coluna_valor], order=(1, 0, 0), exog=df[['intervencao']])
        print("\nSérie estacionária - Modelo ARIMA (1, 0, 0) será usado.")
    else:
        modelo = sm.tsa.ARIMA(df[coluna_valor], order=(1, 1, 1), exog=df[['intervencao']])
        print("\nSérie não estacionária - Modelo ARIMA (1, 1, 1) será usado.")
    
    resultado = modelo.fit()
    
    # Imprimir resumo do modelo
    print(resultado.summary())
    
    # Fazer previsões
    previsao = resultado.predict(start=1, end=len(df)-1, exog=df[['intervencao']].iloc[1:])  # start ajustado
    
    # Plotar os resultados
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[coluna_valor], label='Série Original')
    plt.plot(df.index[1:], previsao, label='Previsão ARIMA com Intervenção', color='red')
    plt.axvline(pd.to_datetime(data_intervencao), color='green', linestyle='--', label='Data de Intervenção')
    plt.legend()
    plt.ylabel(eixo_y)
    title += '- Modelo de Intervenção em Série Temporal'
    plt.title(title)
    plt.show()
    
    # Extrair o resultado e salvar em excel 
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
    
    with pd.ExcelWriter(nome_arquivo) as writer:
        coef_table.to_excel(writer, sheet_name='Coeficientes')
        info_table.to_excel(writer, sheet_name='Informações_Gerais')
        df_previsao.to_excel(writer, sheet_name='Previsoes')
    
    return resultado










os.system('cls')

#Datas de início e fim da pandemia (dia em que foi decretada pelo gov. brasileiro)
datas_marcadas = ['2020-03-06', '2022-05-23'] 


# Inicio da pandemia no Brasil 06/03/2020
# Decreto fim da pandemia no Brasil = 23/05/2022 (fim do estado de Emergência em Saúde Pública de Importância Nacional (Espin)
# Fim da pandemia para OMS 08/05/2023








# Entrada dos Dados ############################################################################

# Dados Abertos- Informações Mercado Imobiliário - BACEN 
# https://dadosabertos.bcb.gov.br/dataset/informacoes-do-mercado-imobiliario
# Baixado CSV - Última Atualização: 15/09/2024
print("Parte 1 - Analisando a Base do Banco Central e Salvando Arquivos Excel das séries de interesse")
nome_arquivo = "1-BACEN-mercadoimobiliario-15092024.csv"


# Relação Indice imobiliário e PIB (série 33)
df_sbpe_33_pib=separa_base_SBPE("indices_imobiliario_pib_br",nome_arquivo)
df_sbpe_33_pib.iloc[:, 0:] = df_sbpe_33_pib.iloc[:, 0:].apply(lambda x: x / 100)


#Valor de compra médio dos imóveis (série 31)
df_sbpe_31_valor_de_compra=separa_base_SBPE(["imoveis_valor_compra_br"],nome_arquivo)


# Área dos Imóveis (série 29)
df_sbpe_29_area=separa_base_SBPE("imoveis_area_privativa_br",nome_arquivo)


#Número de Dormitórios (série 28)
df_sbpe_28_dormitorios=separa_base_SBPE(["imoveis_dormitorio_1_br","imoveis_dormitorio_2_br","imoveis_dormitorio_3_br","imoveis_dormitorio_4_mais_br"],nome_arquivo)


# Tipo de Imóveis (casa e apartamento) (série 27)
df_sbpe_27_tipoimovel=separa_base_SBPE(["imoveis_tipo_apartamento_br","imoveis_tipo_casa_br"],nome_arquivo)
df_sbpe_27_tipoimovel['soma_imoveis'] = df_sbpe_27_tipoimovel['imoveis_tipo_apartamento_br'] + df_sbpe_27_tipoimovel['imoveis_tipo_casa_br']


#Inadinplência por tipo de financiamento (série 19)
df_sbpe_19_inadimplencia=separa_base_SBPE(["credito_estoque_inadimplencia_pf_comercial_br","credito_estoque_inadimplencia_pf_fgts_br","credito_estoque_inadimplencia_pf_home_equity_br","credito_estoque_inadimplencia_pf_livre_br","credito_estoque_inadimplencia_pf_sfh_br"],nome_arquivo)
df_sbpe_19_inadimplencia_agrupada = df_sbpe_19_inadimplencia[["credito_estoque_inadimplencia_pf_fgts_br", "credito_estoque_inadimplencia_pf_sfh_br"]].copy()
df_sbpe_19_inadimplencia_agrupada['outros']=df_sbpe_19_inadimplencia[["credito_estoque_inadimplencia_pf_comercial_br","credito_estoque_inadimplencia_pf_home_equity_br","credito_estoque_inadimplencia_pf_livre_br"]].mean(axis=1)
df_sbpe_19_inadimplencia_agrupada['Data'] = df_sbpe_19_inadimplencia.index
df_sbpe_19_inadimplencia_agrupada=df_sbpe_19_inadimplencia_agrupada[['Data',"credito_estoque_inadimplencia_pf_fgts_br", "credito_estoque_inadimplencia_pf_sfh_br","outros"]]
df_sbpe_19_inadimplencia_agrupada.iloc[:, 1:] = df_sbpe_19_inadimplencia_agrupada.iloc[:, 1:].apply(lambda x: x / 100)
df_sbpe_19_inadimplencia_agrupada.set_index('Data', inplace=True)


# Taxa Contratada (série 11)
df_sbpe_11_taxa_contratada=separa_base_SBPE(["credito_contratacao_taxa_pf_comercial_br","credito_contratacao_taxa_pf_fgts_br","credito_contratacao_taxa_pf_home_equity_br","credito_contratacao_taxa_pf_livre_br","credito_contratacao_taxa_pf_sfh_br"],nome_arquivo)
df_sbpe_11_taxa_contratada_agrupada = df_sbpe_11_taxa_contratada[['credito_contratacao_taxa_pf_fgts_br', 'credito_contratacao_taxa_pf_sfh_br']].copy()
df_sbpe_11_taxa_contratada_agrupada['outros']=df_sbpe_11_taxa_contratada[["credito_contratacao_taxa_pf_comercial_br","credito_contratacao_taxa_pf_home_equity_br","credito_contratacao_taxa_pf_livre_br"]].mean(axis=1)
df_sbpe_11_taxa_contratada_agrupada['Data'] = df_sbpe_11_taxa_contratada.index
df_sbpe_11_taxa_contratada_agrupada=df_sbpe_11_taxa_contratada_agrupada[['Data','credito_contratacao_taxa_pf_fgts_br', 'credito_contratacao_taxa_pf_sfh_br','outros']]
df_sbpe_11_taxa_contratada_agrupada.iloc[:, 1:] = df_sbpe_11_taxa_contratada_agrupada.iloc[:, 1:].apply(lambda x: x / 100)
df_sbpe_11_taxa_contratada_agrupada.set_index('Data', inplace=True)


# Indexador contratado (série 10)
df_sbpe_10_indexador=separa_base_SBPE(["credito_contratacao_indexador_pf_ipca_br","credito_contratacao_indexador_pf_outros_br","credito_contratacao_indexador_pf_prefixado_br","credito_contratacao_indexador_pf_tr_br"],nome_arquivo)
df_sbpe_10_indexador['Total']=df_sbpe_10_indexador[["credito_contratacao_indexador_pf_ipca_br","credito_contratacao_indexador_pf_outros_br","credito_contratacao_indexador_pf_prefixado_br","credito_contratacao_indexador_pf_tr_br"]].sum(axis=1)
df_sbpe_10_indexador.iloc[:, 0:] = df_sbpe_10_indexador.iloc[:, 0:].apply(lambda x: x / 1_000_000_000)
df_sbpe_10_indexador_anual = df_sbpe_10_indexador.resample('YE').sum()


# Modalidade financiada (volume financeiro por modalidade) (série 9)
df_sbpe_09_modalidade=separa_base_SBPE(["credito_contratacao_contratado_pf_comercial_br","credito_contratacao_contratado_pf_home_equity_br","credito_contratacao_contratado_pf_livre_br","credito_contratacao_contratado_pf_sfh_br","credito_contratacao_contratado_pf_fgts_br"],nome_arquivo)
df_sbpe_09_modalidade_agrupada = df_sbpe_09_modalidade[['credito_contratacao_contratado_pf_sfh_br', 'credito_contratacao_contratado_pf_fgts_br']].copy()
df_sbpe_09_modalidade_agrupada['outros'] = df_sbpe_09_modalidade[['credito_contratacao_contratado_pf_comercial_br','credito_contratacao_contratado_pf_home_equity_br','credito_contratacao_contratado_pf_livre_br']].sum(axis=1)
df_sbpe_09_modalidade_agrupada['Data'] = df_sbpe_09_modalidade.index
df_sbpe_09_modalidade_agrupada['total_contratado'] = df_sbpe_09_modalidade_agrupada[['credito_contratacao_contratado_pf_sfh_br','credito_contratacao_contratado_pf_fgts_br','outros']].sum(axis=1)
df_sbpe_09_modalidade_agrupada = df_sbpe_09_modalidade_agrupada[['Data','credito_contratacao_contratado_pf_sfh_br','credito_contratacao_contratado_pf_fgts_br', 'outros','total_contratado']]
df_sbpe_09_modalidade_agrupada.iloc[:, 1:] = df_sbpe_09_modalidade_agrupada.iloc[:, 1:].apply(lambda x: x / 1_000_000_000)
df_sbpe_09_modalidade_agrupada.set_index('Data', inplace=True)
df_sbpe_09_modalidade_anual = df_sbpe_09_modalidade_agrupada.resample('YE').sum()
df_sbpe_09_modalidade_anual.reset_index(inplace=True) # Resetando o índice para que a coluna de data volte a ser uma coluna normal (se necessário)



# Plotando os Gráficos #############################################################################################################################
""""
print('Parte 2 - Plotando os Gráficos de Interesse')
serie = pd.DataFrame(df_sbpe_33_pib)
x_label =''
y_label ='(%)'
title='Relação Mercado Imobiliário e PIB'
plot_series(serie, x_label, y_label, title,datas_marcadas)
"""

################################### DADOS CONSTRUÇÃO #######################################################################################
# Dados Abertos- Custo de Construção por categoria  - CBIC - Camara Brasileira da Industria da construção civil
# http://www.cbicdados.com.br/menu/custo-da-construcao/cub-medio-brasil-custo-unitario-basico-de-construcao-por-m2
# Baixado Excel - Última Atualização: 20/09/2024

nome_arquivo = "2 - CUB 20092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_cub = pd.read_excel(nome_arquivo, sheet_name='tabela_06.A.15')
df_cub.index=pd.to_datetime(df_cub.index)




################################### DADOS COMPRA E ALUGUEL #######################################################################################
# Dados Abertos FIPE-ZAP
# https://www.fipe.org.br/pt-br/indices/fipezap#indice-mensal
# Última Atualização 20/09/2024

nome_arquivo = "3 - fipezap-serieshistoricas-20092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_fipezap = pd.read_excel(nome_arquivo, sheet_name='Preços_FipeZAP')
#df_fipezap['Data'] = pd.to_datetime(df_fipezap['Data'])
df_fipezap.index=pd.to_datetime(df_fipezap['Data'])

################################### DADOS RENDA POPULACIONAL #######################################################################################
# Fonte: PNAD - IBGE 
# https://www.ibge.gov.br/estatisticas/multidominio/genero/9171-pesquisa-nacional-por-amostra-de-domicilios-continua-mensal.html?=&t=series-historicas
# Última Atualização 20/09/2024

nome_arquivo = "4 - PNAD-20092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_pnad = pd.read_excel(nome_arquivo, sheet_name='PNAD')


# Aplicando a função
df_pnad['Data'] = df_pnad['Periodo'].apply(extrair_data_final)
df_pnad=df_pnad[['Data','Renda Média','Taxa de Desocupação','Periodo']]


################################### DADOS DE ECONÔMICOS ########################################################################################

# Reajuste de contratos de aluguel normalmente é utilizado o IGP-M
#Composição:
#60% Índice de Preços ao Produtor Amplo (IPA);
#30%  Índice de Preços ao Consumidor (IPC);
#10%  Índice Nacional de Custo da Construção (INCC).
#IGP-M: https://portal.fgv.br/noticias/igp-m-resultados-2024
# Última Atualização: 27/09/2024

nome_arquivo = "5 - IGPM-FGV-27092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_igpm = pd.read_excel(nome_arquivo, sheet_name='Plan1')


# INPC - variação dos preços para famílias com renda de 1 a 5 salários mínimos
# https://sidra.ibge.gov.br/pesquisa/snipc/inpc/tabelas/brasil/agosto-2024
nome_arquivo = "6 - tabela1736-INPC - 27082024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_inpc = pd.read_excel(nome_arquivo, sheet_name='INPC')


# IPCA - variação de preços todas as rendas
# https://sidra.ibge.gov.br/pesquisa/snipc/ipca/tabelas/brasil/agosto-2024
nome_arquivo = "7 - tabela1737-IPCA-27092024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_ipca = pd.read_excel(nome_arquivo, sheet_name='IPCA')


# Salário Mínimo
#  
nome_arquivo = "8 - Salario - 072024.xlsx"
# Excel já está em uma formatação OK para uso. sem tratamentos adicionais necessários. 
#somente carregamento para posterior utilização
df_salario = pd.read_excel(nome_arquivo, sheet_name='Salário Minimo')




##############################################################################################################################

""""" Data frames processados
df_sbpe_33_pib
df_sbpe_31_valor_de_compra
df_sbpe_29_area
df_sbpe_28_dormitorios
df_sbpe_27_tipoimovel
df_sbpe_19_inadimplencia_agrupada
df_sbpe_11_taxa_contratada_agrupada
df_sbpe_10_indexador
df_sbpe_10_indexador_anual
df_sbpe_09_modalidade_anual
df_sbpe_09_modalidade_agrupada

df_cub 
df_fipezap 

df_pnad

df_igpm
df_inpc
df_ipca

df_salario
"""


# Resetando o índice para que a coluna de data volte a ser uma coluna normal (se necessário)
df_sbpe_33_pib.reset_index(inplace=True)
df_sbpe_31_valor_de_compra.reset_index(inplace=True)
df_sbpe_29_area.reset_index(inplace=True)
df_sbpe_28_dormitorios.reset_index(inplace=True)
df_sbpe_27_tipoimovel.reset_index(inplace=True)
df_sbpe_19_inadimplencia_agrupada.reset_index(inplace=True)
df_sbpe_11_taxa_contratada_agrupada.reset_index(inplace=True)
df_sbpe_10_indexador.reset_index(inplace=True)
df_sbpe_09_modalidade_agrupada.reset_index(inplace=True)

dataframes={'df_sbpe_33_pib':df_sbpe_33_pib,
            'df_sbpe_31_valor_de_compra':df_sbpe_31_valor_de_compra,
            'df_sbpe_29_area':df_sbpe_29_area,
            'df_sbpe_28_dormitorios':df_sbpe_28_dormitorios,
            'df_sbpe_27_tipoimovel':df_sbpe_27_tipoimovel,
            'df_sbpe_19_inadimplencia':df_sbpe_19_inadimplencia_agrupada,
            'df_sbpe_11_taxa_contratada':df_sbpe_11_taxa_contratada_agrupada,
            'df_sbpe_10_indexador':df_sbpe_10_indexador,
            'df_sbpe_10_indexador_anual':df_sbpe_10_indexador_anual,
            'df_sbpe_09_modalidade_anual':df_sbpe_09_modalidade_anual,
            'df_sbpe_09_modalidade_agrupada':df_sbpe_09_modalidade_agrupada,
            'df_cub':df_cub,
            'df_fipezap':df_fipezap,
            'df_pnad':df_pnad,
            'df_igpm':df_igpm,
            'df_inpc':df_inpc,
            'df_ipca':df_ipca,
            'df_salario':df_salario
            }

salva_dfs_em_excel(dataframes, '9-processado')

""""
verificar_estacionaridade(df_sbpe_33_pib, ['indices_imobiliario_pib_br'])
verificar_estacionaridade(df_sbpe_31_valor_de_compra,['imoveis_valor_compra_br'])
verificar_estacionaridade(df_sbpe_27_tipoimovel,['soma_imoveis'])
verificar_estacionaridade(df_sbpe_09_modalidade_agrupada,['total_contratado'])
verificar_estacionaridade(df_cub, ['Global'])
verificar_estacionaridade(df_fipezap, ['Preço Médio Venda'])
verificar_estacionaridade(df_fipezap, ['Preço Médio Aluguel'])
verificar_estacionaridade(df_pnad, ['Renda Média'])
"""


####### Análise da Relação PIB e Mercado imobiliário 

title='Relação Mercado imobiliário e PIB '
eixo_y=''
nome_arquivo='10-SARIMA_PIB.xlsx'
decompoe_serie_unica(df_sbpe_33_pib, ['Data'], ['indices_imobiliario_pib_br'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_sbpe_33_pib, 'Data', 'indices_imobiliario_pib_br', '2020-03-06',nome_arquivo,title,eixo_y)

####### Análise do valor de compra 

title='Valor de Compra dos imóveis financiados '
eixo_y='R$'
nome_arquivo='10-SARIMA_financiado_valorcompra.xlsx'
decompoe_serie_unica(df_sbpe_31_valor_de_compra, ['Data'], ['imoveis_valor_compra_br'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_sbpe_31_valor_de_compra, 'Data', 'imoveis_valor_compra_br', '2020-03-06',nome_arquivo,title,eixo_y)

###### Análise da quantidade de negócios realizados 

title='Número de negócios realizados'
eixo_y=''
nome_arquivo='10-SARIMA_numero_negocios.xlsx'
decompoe_serie_unica(df_sbpe_27_tipoimovel, ['Data'], ['soma_imoveis'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_sbpe_27_tipoimovel, 'Data', 'soma_imoveis', '2020-03-06',nome_arquivo,title,eixo_y)


####### Análise do volume financeiro total dos financiamentos

title='Volume financeiro total dos financiamentos'
eixo_y='em Bilhões R$'
nome_arquivo='10-SARIMA_volumefinanciado.xlsx'
decompoe_serie_unica(df_sbpe_09_modalidade_agrupada, ['Data'], ['total_contratado'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_sbpe_09_modalidade_agrupada, 'Data', 'total_contratado', '2020-03-06',nome_arquivo,title,eixo_y)


####### Análise do custo de construção

title='Custo de construção (CUB)'
eixo_y='R$/m2'
nome_arquivo='10-SARIMA_CUB.xlsx'
decompoe_serie_unica(df_cub, ['Data'], ['Global'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_cub, 'Data', 'Global', '2020-03-06',nome_arquivo,title,eixo_y)


####### Análise do valor de venda dos imóveis conforme FIPEZAP

title='Preço de Venda dos Imóveis'
eixo_y='R$/m2'
nome_arquivo='10-SARIMA_fipezap_compra.xlsx'
decompoe_serie_unica(df_fipezap, ['Data'], ['Preço Médio Venda'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_fipezap, 'Data', 'Preço Médio Venda', '2020-03-06',nome_arquivo,title,eixo_y)


####### Análise do valor de aluguel dos imóveis conforme FIPEZAP

title='Preço de Aluguel dos Imóveis'
eixo_y='R$/m2'
nome_arquivo='10-SARIMA_fipezap_aluguel.xlsx'
decompoe_serie_unica(df_fipezap, ['Data'], ['Preço Médio Aluguel'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_fipezap, 'Data', 'Preço Médio Aluguel', '2020-03-06',nome_arquivo,title,eixo_y)

####### Análise da Renda da população
title='Renda Média da População'
eixo_y='R$'
nome_arquivo='10-SARIMA_pnad_renda.xlsx'
decompoe_serie_unica(df_pnad, ['Data'], ['Renda Média'], title, datas_marcadas, eixo_y)

resultado = modelo_intervencao(df_pnad, 'Data', 'Renda Média', '2020-03-06',nome_arquivo,title,eixo_y)



