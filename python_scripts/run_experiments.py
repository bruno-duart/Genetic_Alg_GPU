import pandas as pd
import subprocess
import os
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
EXECUTABLE = "./main"
INSTANCES_FILE = "info/instances_with_k.csv"
OUTPUT_FILE = "resultados/resultados_gpu.csv"
REPETITIONS = 30  # Quantas vezes rodar cada instância para ter média estatística
INSTANCES_DIR = "instances/" # Onde estão os arquivos .col
TIMEOUT_SEC = 180

def run_single_experiment(instance_path, k, seed):
    """
    Roda o executável CUDA C++ e captura a saída
    
    :param instance_path: Path para a instância a ser executada
    :param k: Número de cores disponíveis
    :param seed: Seed aleatória
    """
    try:
        # Chama o C++
        result = subprocess.run(
            [EXECUTABLE, instance_path, str(k), str(seed)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC
        )
        
        # Procura o resultado
        for line in result.stdout.splitlines():
            if line.startswith("CSV_RESULT;"):
                parts = line.split(";")
                return {
                    "status": "success",
                    "fitness": int(parts[1]),
                    "time": float(parts[2]),
                    "generations": int(parts[3])
                }
                
        return {"status": "error", "error_msg": "Output format mismatch"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "fitness": np.nan, "time": float(TIMEOUT_SEC) }
    except Exception as e:
        return {"status": "crash", "error_msg": str(e)}
    
def load_existing_progress():
    """
    Lê o CSV de resultados (se existir) e cria um conjunto de chaves 
    (instância, repetição) que já foram concluídas
    """    
    finished = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            # Lê apenas as colunas necessárias para verificar duplicatas
            df = pd.read_csv(OUTPUT_FILE, sep=";", usecols=["instance", "run_id"])
            for _, row in df.iterrows():
                finished.add((row['instance'], row['run_id']))
        except Exception as e:
            print(f"Aviso: Não foi possível ler progresso anterior ({e}). Começando do zero.")
    return finished

def main():
    # Carrega lista de instâncias
    try: 
        df_inst = pd.read_csv(INSTANCES_FILE, sep=";")
        print(f"Carregadas {len(df_inst)} instâncias. Meta: {REPETITIONS} reps cada.")
    except Exception as e:
        print(f"Erro ao ler {INSTANCES_FILE}: {e}")
        return
    
    # Prepara arquivo de saída
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("timestamp;instance;k;run_id;seed;fitness;time_sec;generations;status\n")
           
    finished_runs = load_existing_progress()
    len_finished = len(finished_runs)
    if len_finished > 0:
        print(f"Já foram concluídas {len_finished} execuções anteriormente.")
         
    total_runs = len(df_inst) * REPETITIONS
    
    with tqdm(total=total_runs, initial=len_finished) as pbar:
        for index, row in df_inst.iterrows():
            inst_name = row['instance_name']
            k_target = row['k']
            
            inst_path = os.path.join(INSTANCES_DIR, f"{inst_name}.col")
            
            if not os.path.exists(inst_path):
                print(f"⚠️ Arquivo não encontrado: {inst_path}. Pulando.")
                pbar.update(REPETITIONS)
                continue
            
            for rep in range(REPETITIONS):
                
                if (inst_name, rep) in finished_runs:
                    continue
                # Gera seed
                seed = np.random.randint(1, 10000000)
                
                res = run_single_experiment(instance_path=inst_path, k=k_target, seed=seed)
                
                with open(OUTPUT_FILE, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fitness = res.get('fitness', -1)
                    time_s = res.get('time', -1)
                    gens = res.get('generations', 0)
                    status = res['status']
                    
                    line = f"{timestamp};{inst_name};{k_target};{rep};{seed};{fitness};{time_s};{gens};{status}\n"
                    f.write(line)
                    f.flush() # Força a gravação no disco
                    
                pbar.update(1)
                
    print("\nExperimentos finalizados!")
    print(f"Resultados salvos em: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()