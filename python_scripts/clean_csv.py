import pandas as pd
import io

# 1. Carrega e corrige o CSV (que tem aquele problema no cabeçalho)
with open('./resultados/resultados_gpu.csv', 'r') as f:
    lines = f.readlines()

# Corrige cabeçalho colado se necessário
if "status202" in lines[0]:
    header = lines[0].strip()
    idx = header.find('status')
    real_header = header[:idx+6]
    first_data = header[idx+6:]
    lines[0] = real_header + "\n"
    lines.insert(1, first_data + "\n")

# Lê os dados ignorando erros de formatação
# on_bad_lines='skip' ignora as linhas corrompidas (timeout/crash)
df = pd.read_csv(io.StringIO("".join(lines)), sep=';', on_bad_lines='skip')

# Converte fitness para número
df['fitness'] = pd.to_numeric(df['fitness'], errors='coerce')

# 2. SEPARA O JOIO DO TRIGO
# Mantemos tudo que NÃO foi sucesso ótimo (ou seja, fitness > 0 ou timeouts que você queira manter)
# Se fitness for NaN (timeout), mantemos também para o python refazer ou não, conforme sua lógica.
# A lógica aqui é: Se fitness == 0, a coluna generations está errada. APAGUE.
df_to_keep = df[df['fitness'] != 0]

print(f"Total original: {len(df)}")
print(f"Mantidos (Difíceis/Corretos): {len(df_to_keep)}")
print(f"Removidos para reprocessar (Fáceis): {len(df) - len(df_to_keep)}")

# 3. Salva o arquivo limpo
df_to_keep.to_csv('resultados/resultados_gpu.csv', sep=';', index=False)
print("Arquivo CSV atualizado com sucesso!")