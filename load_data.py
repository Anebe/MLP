import csv
from datetime import datetime

# Abrindo o arquivo CSV e lendo os dados
def read():
    input_data = []
    output_data = []

    with open("dataset_pedidos.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_str = row["DATA_ENTREGA"]
            
            data_obj = datetime.strptime(data_str, "%Y-%m-%d")
            #timestamp = data_obj.timestamp()
            #timestamp = scale_timestamps(timestamp)
            dia = data_obj.day
            mes = data_obj.month
            ano = data_obj.year

            temperatura = float(row["TEMPERATURA"])/100
            
            quantidade = int(row["QUANTIDADE_PEDIDA_NO_DIA"])/100

            # Adicionando aos arrays
            input_data.append([temperatura])
            output_data.append(quantidade)
        
        output_data = [[x] for x in output_data]

        return input_data, output_data

# Abrindo o arquivo CSV e lendo os dados
def read_walmart():
    input_data = []
    output_data = []

    with open("walmart.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_str = row["Date"]
            
            data_obj = datetime.strptime(data_str, "%d-%m-%Y")
            #timestamp = data_obj.timestamp()
            #timestamp = scale_timestamps(timestamp)
            dia = data_obj.day
            mes = data_obj.month
            ano = data_obj.year

            temperatura = float(row["Temperature"])
            
            quantidade = float(row["Weekly_Sales"])
            quantidade = int(str(int(quantidade))[:3])

            # Adicionando aos arrays
            input_data.append([dia, mes, ano, temperatura])
            output_data.append(quantidade)
        
        #output_data = max_min(output_data)
        #input_data = max_min(input_data)
        #out_copy = output_data.copy()
        #for i in range(len(output_data)):
        #    output_data[i] = (out_copy[i] - min(out_copy)) / (max(out_copy) - min(out_copy))
        output_data = [[x] for x in output_data]

        return input_data, output_data

def max_min(array):
    array_copy = array.copy()
    for i in range(len(array)):
        array[i] = (array_copy[i] - min(array_copy)) / (max(array_copy) - min(array_copy))
    return array

# Abrindo o arquivo CSV e lendo os dados
def read_speed():
    input_data = []
    output_data = []
    distancias = []
    tempos = []
    with open("dataset_velocidade.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            distancia = float(row["Distancia"])
            tempo = float(row["Tempo"])
            velocidade = float(row["Velocidade"])

            # Adicionando aos arrays
            #distancias.append(distancia)
            #tempos.append(tempo)
            
            input_data.append([distancia,tempo])
            output_data.append(velocidade)
        
        #distancias = max_min(distancias)
        #tempos = max_min(tempos)
        #input_data = [[distancias[i], tempos[i]] for i in range(len(distancias))]
        #output_data = max_min(output_data)
        
        output_data = [[x] for x in output_data]

        return input_data, output_data

def scale_timestamps(timestamps):
    # Reduz a escala dividindo pelo número de segundos em um dia
    seconds_in_a_day = 86400  # 24 * 60 * 60
    return timestamps / seconds_in_a_day

def split_array(arr, percentage):
    # Calcula o índice para a separação
    split_index = int(len(arr) * percentage)
    
    # Separa o array
    part1 = arr[:split_index]
    part2 = arr[split_index:]
    
    return part1, part2