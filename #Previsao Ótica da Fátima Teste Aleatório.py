import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import calendar as cal
# Carregar dados
data = pd.read_csv('C:/Users/ivanl/OneDrive/Área de Trabalho/Loja/Valor por Data.csv', sep=';')
data = data.dropna()

# Converter tipos
data['SEMANA'] = data['SEMANA'].astype(int)
data['DATAMOVIMENTO'] = pd.to_datetime(data['DATAMOVIMENTO'], dayfirst=True, format='%d/%m/%Y')

# Extrair features de data
data['DIA'] = data['DATAMOVIMENTO'].dt.day
data['MES'] = data['DATAMOVIMENTO'].dt.month
data['ANO'] = data['DATAMOVIMENTO'].dt.year
data['DIA_SEMANA'] = data['DATAMOVIMENTO'].dt.dayofweek

# Preparar variáveis
x = data[['SEMANA', 'DIA', 'MES', 'ANO', 'DIA_SEMANA']]
y = data.iloc[:, 2].str.replace(',', '.').astype(float)  # Assumindo que a coluna 2 é o target

# Dividir dados
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=99)

# Treinar modelo
reg = LinearRegression()
reg.fit(x_train, y_train)

# Prever e mostrar resultados
y_pred = reg.predict(x_test)
results = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})

def prever_para_data():
    # Solicitar data do usuário
    data_input = input('Digite a data (dd/mm/aaaa): ')
    
    try:
        # Converter a string para datetime
        data_dt = pd.to_datetime(data_input, dayfirst=True, format='%d/%m/%Y')
        
        # Extrair features no mesmo formato do treino
        semana = data_dt.isocalendar().week
        dia = data_dt.day
        mes = data_dt.month
        ano = data_dt.year
        dia_semana = data_dt.dayofweek
        
        # Criar DataFrame no formato correto
        dados_entrada = pd.DataFrame({
            'SEMANA': [semana],
            'DIA': [dia],
            'MES': [mes],
            'ANO': [ano],
            'DIA_SEMANA': [dia_semana]
        })
        
        # Fazer a previsão
        previsao = reg.predict(dados_entrada)
        
        # Mostrar resultado
        dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        print(f"\nPrevisão para {data_input} ({dias_semana[dia_semana]}):")
        print(f"Valor previsto: R$ {previsao[0]:.2f}")
        
        return previsao[0]
    
    except ValueError:
        print("Formato de data inválido! Use dd/mm/aaaa")
        return None

# Métricas de avaliação
from sklearn.metrics import mean_squared_error, r2_score
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt
# INICIO DO TESTE
bins = np.quantile(y_test, [0, 0.25, 0.5, 0.75, 1])
colors = np.digitize(y_test, bins)
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred , alpha=0.5, c=colors,cmap='rainbow')
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Comparação entre Valores Reais e Previstos")
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(reg, x, y, cv=5)
print("Accuracy:", scores.mean())
print("Desvio padrão:", scores.std())


def previsao_media_mes():
    meses = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    
    print("\nMeses disponíveis:")
    for num, nome in meses.items():
        print(f"{num} - {nome}")
    
    try:
        mes = int(input("\nEscolha o número do mês: "))
        ano = int(input("Digite o ano (ex: 2023): "))
        
        if mes not in meses: 
            raise ValueError
        
        # Gerar todas as datas do mês
        datas = pd.date_range(start=f"{ano}-{mes}-01", end=f"{ano}-{mes}-28", freq='D')  # Até dia 28 para simplificar
        
        # Preparar dados para previsão
        dados = pd.DataFrame({
            'SEMANA': [d.isocalendar().week for d in datas],
            'DIA': [d.day for d in datas],
            'MES': [d.month for d in datas],
            'ANO': [d.year for d in datas],
            'DIA_SEMANA': [d.dayofweek for d in datas]
        })
        
        # Fazer previsões
        previsoes = reg.predict(dados)
        media = np.mean(previsoes)*int(cal.monthrange(ano, mes)[1])
        
        print(f"\nPrevisão média para {meses[mes]}/{ano}: R$ {media:.2f}")
        return media
        
    except ValueError:
        print("Valor inválido! Digite um mês (1-12) e ano válidos.")
        return None

def previsao_media_semana():
    try:
        data_input = input("Digite a data inicial (dd/mm/aaaa): ")
        data_inicial = pd.to_datetime(data_input, dayfirst=True, format='%d/%m/%Y')
        
        # Gerar os 7 dias seguintes
        datas = pd.date_range(start=data_inicial, periods=7)
        
        # Preparar dados para previsão
        dados = pd.DataFrame({
            'SEMANA': [d.isocalendar().week for d in datas],
            'DIA': [d.day for d in datas],
            'MES': [d.month for d in datas],
            'ANO': [d.year for d in datas],
            'DIA_SEMANA': [d.dayofweek for d in datas]
        })
        
        # Fazer previsões
        previsoes = reg.predict(dados)
        
        # Mostrar resultados detalhados
        dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        print("\nPrevisões diárias:")
        for i, (data, valor) in enumerate(zip(datas, previsoes)):
            print(f"{data.strftime('%d/%m/%Y')} ({dias_semana[data.dayofweek]}): R$ {valor:.2f}")
        
        media = np.mean(previsoes)*7
        print(f"\nMédia para os 7 dias: R$ {media:.2f}")
        return media
        
    except ValueError:
        print("Data inválida! Use o formato dd/mm/aaaa.")
        return None
    
def menu_principal():
    while True:
        print("\n==== MENU DE PREVISÕES ====")
        print("1 - Previsão para uma data específica")
        print("2 - Previsão média por mês")
        print("3 - Previsão média para uma semana")
        print("4 - Sair")
        
        opcao = input("\nEscolha uma opção: ")
        
        if opcao == '1':
            prever_para_data()
        elif opcao == '2':
            previsao_media_mes()
        elif opcao == '3':
            previsao_media_semana()
        elif opcao == '4':
            print("Saindo do programa...")
            break
        else:
            print("Opção inválida! Digite 1-4.")

menu_principal()