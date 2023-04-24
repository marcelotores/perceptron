import pandas as pd

dados = 'dados/dados_classificacao2.csv'

dados_df = pd.read_csv(dados)

print(dados_df)

numero_atributo_por_classe = dados_df.groupby(['classe'])['classe'].count()

print(numero_atributo_por_classe)

