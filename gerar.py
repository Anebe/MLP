import pandas as pd
import random

# Gerar dataset com valores inteiros de distância (d) e tempo (t)
data = []
for _ in range(1000):
    d = random.randint(1, 100)  # Distância aleatória entre 1 e 100
    t = random.randint(1, 10)   # Tempo aleatório entre 1 e 10
    v = round(d / t, 2)                   # Calcular a velocidade v
    data.append([d, t, v])

# Criar o DataFrame
df = pd.DataFrame(data, columns=["Distancia", "Tempo", "Velocidade"])

# Salvar o DataFrame em um arquivo CSV
csv_file_path = "dataset_velocidade.csv"
df.to_csv(csv_file_path, index=False)

csv_file_path
