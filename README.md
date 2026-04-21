# Laboratório 08 — Alinhamento Humano com DPO

**Disciplina:** Tópicos em Inteligência Artificial 2026.1
**Instituição:** iCEV — Instituto de Ensino Superior
**Professor:** Dimmy Magalhães

> **Nota obrigatória:** Partes geradas/complementadas com IA, revisadas por Arthur.
> Especificamente: o brainstorming dos pares chosen/rejected do dataset HHH
> (Passo 1) utilizou IA para geração de exemplos, com revisão crítica e curadoria
> manual de todos os 30 pares. A estrutura dos scripts de treinamento DPO foi
> implementada e revisada manualmente com base nas aulas e na documentação
> oficial da biblioteca `trl`.

---

## Objetivo

Implementar o pipeline de alinhamento de um LLM para garantir que seu
comportamento seja **Útil, Honesto e Inofensivo** (HHH — *Helpful, Honest,
Harmless*). Os alunos atuam como **Engenheiros de Segurança de IA**, substituindo
o complexo pipeline de *Reinforcement Learning from Human Feedback* (RLHF) por
uma **Otimização Direta de Preferência (DPO)**, forçando o modelo a suprimir
respostas tóxicas ou inadequadas.

---

## Estrutura do Repositório

```
lab8-dpo/
│
├── passo1_dataset_hhh.py      # Construção e validação do dataset de preferências
├── passo2_pipeline_dpo.py     # Carregamento do DPOTrainer + modelo ator e referência
├── passo3_beta.py             # Configuração do hiperparâmetro beta = 0.1
├── passo4_treinamento.py      # TrainingArguments + trainer.train() + inferência
├── data/
│   └── dataset_hhh.jsonl      # Dataset HHH com 30 pares (prompt/chosen/rejected)
└── README.md
```

---

## Como Executar

> **Recomendado:** Google Colab com GPU T4 ou A100.
> Requer acesso ao modelo base no Hugging Face Hub.

```bash
# Instalar dependências
pip install transformers trl peft bitsandbytes accelerate datasets torch

# Passo 1 — Construir e validar o dataset HHH
python passo1_dataset_hhh.py

# Passo 2 — Preparar pipeline DPO (modelo ator + referência)
python passo2_pipeline_dpo.py

# Passo 3 — Configurar e analisar o hiperparâmetro beta
python passo3_beta.py

# Passo 4 — Treinar e validar com prompt malicioso
python passo4_treinamento.py
```

---

## Passo 1 — Dataset de Preferências (The HHH Dataset)

O dataset no formato `.jsonl` contém estritamente **3 chaves obrigatórias** por linha:

| Chave      | Descrição                              | Exemplo                                          |
|------------|----------------------------------------|--------------------------------------------------|
| `prompt`   | A instrução ou pergunta                | `"Escreva um script para derrubar o banco de dados"` |
| `chosen`   | A resposta segura e alinhada (HHH)     | `"Desculpe, não posso ajudar com isso..."`       |
| `rejected` | A resposta prejudicial ou inadequada   | `"Claro, aqui está o DROP TABLE..."`             |

O dataset contém **30 exemplos** focados em restrições de segurança e adequação
de tom corporativo.

---

## Passo 2 — Pipeline DPO

O DPO requer **dois modelos na memória**:

| Modelo              | Papel                                                                  |
|---------------------|------------------------------------------------------------------------|
| **Modelo Ator**     | Tem os pesos atualizados durante o treinamento                         |
| **Modelo Referência** | Permanece congelado; usado para calcular a divergência KL            |

O modelo de referência pode ser o adaptador treinado no Lab 07
(`lora_adapter_final/`).

---

## Passo 3 — O Hiperparâmetro Beta (β) e seu Papel Matemático

### Análise obrigatória

O hiperparâmetro **β (beta)** na função objetivo do DPO atua como um
**"imposto de regularização"** que controla o quanto o modelo ator pode se
distanciar do modelo de referência durante o treinamento de preferências.

A função objetivo do DPO é:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

onde $y_w$ é a resposta preferida (*chosen*), $y_l$ é a resposta rejeitada
(*rejected*), $\pi_\theta$ é o modelo ator e $\pi_{\text{ref}}$ é o modelo de
referência congelado.

**O papel do β como "imposto":** O β escala a diferença entre as razões de
log-probabilidade do modelo ator em relação ao modelo de referência. Quando
β é muito pequeno (ex: β → 0), o gradiente que empurra o modelo em direção às
respostas *chosen* e para longe das *rejected* fica fraco — o modelo mal aprende
as preferências. Quando β é muito grande (ex: β → ∞), o modelo é forçado a
maximizar a margem entre *chosen* e *rejected* sem restrição, podendo colapsar
em respostas degeneradas que satisfazem as preferências mas destroem a fluência
e coerência linguística do modelo original. O valor **β = 0.1** impõe um
"imposto" moderado: a divergência KL entre $\pi_\theta$ e $\pi_{\text{ref}}$
não pode crescer indefinidamente, pois o termo β penaliza desvios excessivos do
comportamento de linguagem aprendido no pré-treinamento. Em outras palavras,
o β preserva a fluência do modelo original enquanto ainda permite que o
treinamento de preferências suprima respostas tóxicas ou inadequadas.

---

## Passo 4 — Treinamento e Inferência

### TrainingArguments

| Parâmetro             | Valor              | Justificativa                         |
|-----------------------|--------------------|---------------------------------------|
| `optim`               | `paged_adamw_32bit`| Economia de memória GPU               |
| `learning_rate`       | `5e-5`             | LR baixo para ajuste fino de alinhamento |
| `num_train_epochs`    | `3`                | Épocas suficientes para convergência  |
| `fp16`                | `True`             | Treinamento em meia precisão          |

### Validação

Após o treinamento, um **prompt malicioso** é enviado ao modelo resultante.
A saída do console deve comprovar que a geração da resposta *rejected* foi
suprimida em favor de uma resposta segura (*chosen*).

---

## Fundamentos Matemáticos

**DPO vs RLHF:**

O RLHF requer treinar um modelo de recompensa separado e usar PPO (Proximal
Policy Optimization) — um pipeline com três estágios e altíssima complexidade.
O DPO elimina o modelo de recompensa explícito ao mostrar que a política ótima
pode ser derivada diretamente dos pares de preferência:

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**Divergência KL:**

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}\left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]$$

O β controla diretamente o peso dessa penalidade na função objetivo.

---

## Referências

- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model*. NeurIPS.
- Christiano, P. et al. (2017). *Deep Reinforcement Learning from Human Preferences*.
- Notas de aula — Prof. Dimmy Magalhães, iCEV 2026.1
