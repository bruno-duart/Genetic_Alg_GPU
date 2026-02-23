# GPU-Accelerated Genetic Algorithm for Graph Coloring Problem (GCP)

Este repositório contém a implementação de um Algoritmo Genético paralelizado em GPU (usando CUDA C++) para resolver o Problema de Coloração de Grafos (GCP). O projeto foi desenvolvido como parte da disciplina Computação em GPU, para o Programa de Pós-Graduação em Engenharia Elétrica e de Computação, focando em explorar o paralelismo massivo da GPU para operadores genéticos, avaliação de fitness e gerenciamento eficiente de memória (formato CSR e Double Buffering).

## 💻 Ambiente de Desenvolvimento

O projeto foi desenvolvido, compilado e testado no seguinte ambiente:
- **SO Hospedeiro:** Windows 11
- **Subsistema Linux:** WSL2 (Ubuntu 22.04 LTS)
- **IDE:** Visual Studio Code (VSCode)
- **Compilador CUDA:** Cuda compilation tools, release 12.9, V12.9.86 (Build cuda_12.9.r12.9/compiler.36037853_0)
- **Hardware GPU:** NVIDIA RTX 3050 6GB - *Recomenda-se o uso do "Modo de Desempenho" para testes longos.*

## 🛠️ Pré-requisitos e Dependências

Para compilar e executar este projeto, você precisará de:

### 1. C++/CUDA
- **NVIDIA CUDA Toolkit (12.9+)**: Essencial para compilação com `nvcc`.
- Compilador C++ com suporte a C++14 ou superior (ex: `g++`).

### Python (Para Automação de Testes)
- Python 3.8+
- Bibliotecas: `pandas`, `numpy`, `tqdm`
  ```bash
  pip install pandas numpy tqdm

  ```

## 🏗️ Compilação do Projeto

O código C++/CUDA deve ser compilado utilizando o `nvcc`. Na raiz do projeto, abra o terminal e execute o seguinte comando:

```bash
nvcc -Wno-deprecated-gpu-targets main.cpp lib/graphs.cpp lib/solution.cpp lib/utils.cpp lib/gpu_translator.cu lib/genetic_gpu.cu -I include/ -o main

```

## 🚀 Como Executar os Testes

Você pode executar o projeto de duas maneiras: avaliando uma instância única ou rodando a bateria completa de testes automatizados.

### Opção 1: Execução Individual (Instância Única)

Ideal para debugar, testar alterações rápidas ou avaliar um grafo específico. O executável requer três argumentos: o caminho do grafo, o número de cores (`k`) e uma semente aleatória (seed).

**Comando:**

```bash
./main <caminho_arquivo.col> <numero_cores> <seed>

```

**Exemplo Prático:**

```bash
./main instances/le450_15a.col 15 123456

```

**Saída Esperada:** O programa imprimirá o progresso (opcional) e finalizará com uma linha no formato CSV contendo os resultados finais:
`CSV_RESULT;<melhor_fitness>;<tempo_em_segundos>;<geracao_de_parada>`

---

### Opção 2: Execução em Lote (Bateria de Testes Automatizados)

Para rodar experimentos massivos em múltiplas instâncias e compilar os resultados de forma segura para análise estatística, utilize o script Python fornecido (`run_experiments.py`).

Este script lê a lista de grafos e o número de cores do arquivo `info/instances_with_k.csv` e executa o binário CUDA 30 vezes para cada instância.

**Comando:**

```bash
python3 python_scrpits/run_experiments.py

```

**Recursos de Resiliência do Script:**

* **Mecanismo de Retomada (Resume):** Se você interromper a execução (`Ctrl+C`) ou o computador reiniciar, basta rodar o comando novamente. O script lerá o arquivo `resultados/resultados_gpu.csv` e continuará exatamente de onde parou, pulando as repetições já concluídas.
* **Controle de Timeout:** Instâncias que excedem o tempo limite de 180s são interrompidas pelo Python (nota: o código C++ possui um mecanismo interno de 170s para salvar resultados parciais antes do timeout brusco do Python).
* **Salvamento em Tempo Real:** Grava os resultados linha a linha (com `flush`), prevenindo perda de dados em caso de falha.


## 📂 Estrutura do Diretório

* `instances/` - Diretório contendo os arquivos de benchmark de grafos no formato DIMACS (`.col`).
* `include/` - Headers C++ e CUDA (`.h`, `.cuh`).
* `lib/` - Implementações dos métodos e Kernels (`.cpp`, `.cu`).
* `info/instances_with_k.csv` - Tabela de instâncias e número de cores base para os testes.
* `resultados/` - Diretório onde os arquivos `.csv` de saída são gerados.
* `main.cpp` - Ponto de entrada do programa C++.
* `python_scripts/run_experiments.py` - Script principal de automação e coleta de dados.
