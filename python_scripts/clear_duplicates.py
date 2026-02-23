import pandas as pd

# Lê o arquivo sujo
df = pd.read_csv("resultados/resultados_gpu.csv", sep=";")

# Remove duplicatas mantendo a última ocorrência (ou primeira)
# Subset define que duplicata é baseada na instância e no número da repetição
df_clean = df.drop_duplicates(subset=['instance', 'run_id'], keep='last')

# Salva de volta
df_clean.to_csv("resultados/resultados_gpu.csv", sep=";", index=False)

print(f"Linhas antes: {len(df)}. Linhas depois: {len(df_clean)}.")
