"""
Laboratório 8 — Passo 3: A Engenharia do Hiperparâmetro Beta
=============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Configura o hiperparâmetro beta = 0.1 no DPOConfig e analisa seu papel
matemático como "imposto" que impede que a Otimização de Preferência
destrua a fluência do modelo de linguagem original.

Análise do papel matemático do beta
-------------------------------------
O beta (β) escala a penalidade da divergência KL entre o modelo ator (π_θ)
e o modelo de referência (π_ref) na função objetivo do DPO:

    L_DPO = -E[ log σ( β * log(π_θ(y_w|x)/π_ref(y_w|x))
                     - β * log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

    β → 0  : penalidade KL fraca — modelo aprende pouco as preferências
    β → ∞  : nenhuma restrição KL — modelo pode colapsar em texto degenerado
    β = 0.1: equilíbrio — suprime respostas rejeitadas sem perder fluência

Dependências
------------
    pip install trl transformers torch
"""

from trl import DPOConfig


# ---------------------------------------------------------------------------
# Passo 3 — DPOConfig com beta = 0.1
# ---------------------------------------------------------------------------

BETA = 0.1


def criar_dpo_config(output_dir="dpo_output"):
    """
    Instancia o DPOConfig com o hiperparâmetro beta = 0.1 conforme
    especificado no enunciado do laboratório.

    O DPOConfig é uma extensão do TrainingArguments com parâmetros
    específicos do algoritmo DPO, incluindo o beta que controla o
    peso da penalidade KL na função objetivo.

    Parâmetros do DPOConfig
    -----------------------
    beta            : 0.1  — "imposto" de regularização KL (obrigatório)
    max_length      : 512  — comprimento máximo das sequências
    max_prompt_length: 128 — comprimento máximo do prompt

    Parâmetros do TrainingArguments (herdados)
    ------------------------------------------
    optim           : paged_adamw_32bit  — economia de memória GPU
    learning_rate   : 5e-5              — LR para alinhamento
    num_train_epochs: 3
    fp16            : True
    logging_steps   : 10

    Retorna
    -------
    dpo_config : DPOConfig
    """
    dpo_config = DPOConfig(
        # --- Hiperparâmetro obrigatório do enunciado ---
        beta             = BETA,
        # -----------------------------------------------
        max_length       = 512,
        max_prompt_length = 128,
        output_dir       = output_dir,
        optim            = "paged_adamw_32bit",
        learning_rate    = 5e-5,
        num_train_epochs = 3,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        fp16             = True,
        logging_steps    = 10,
        save_steps       = 50,
        save_total_limit = 2,
        remove_unused_columns = False,
        report_to        = "none",
    )
    return dpo_config


# ---------------------------------------------------------------------------
# Análise do impacto do beta na função objetivo
# ---------------------------------------------------------------------------

def analisar_beta(betas=None):
    """
    Demonstra numericamente o impacto de diferentes valores de beta
    na função objetivo do DPO, mostrando o comportamento de "imposto".

    Para simplificar, assume log-ratios fixos e varia apenas o beta.
    """
    import math

    if betas is None:
        betas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    # Log-ratios simulados (diferença entre chosen e rejected)
    # Um valor positivo significa que o modelo prefere a resposta chosen
    log_ratio_chosen   =  0.8   # π_θ(y_w|x) / π_ref(y_w|x)
    log_ratio_rejected = -0.5   # π_θ(y_l|x) / π_ref(y_l|x)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    print(f"\n{'Beta':>8} | {'Argumento σ':>14} | {'σ(·) — prob. escolha certa':>26} | {'Gradiente':>12}")
    print("-" * 70)

    for beta in betas:
        arg        = beta * (log_ratio_chosen - log_ratio_rejected)
        prob       = sigmoid(arg)
        grad_proxy = prob * (1 - prob)   # derivada da sigmoid ≈ força do gradiente
        flag       = " ← beta=0.1 (enunciado)" if beta == 0.1 else ""
        print(f"  {beta:>6.2f}  |  {arg:>12.4f}  |  {prob:>24.6f}  |  {grad_proxy:>10.6f}{flag}")

    print("\nInterpretação:")
    print("  β muito pequeno → argumento σ próximo de 0 → gradiente forte mas")
    print("                    sinal de preferência fraco → modelo aprende devagar")
    print("  β muito grande  → argumento σ muito alto → σ saturada → gradiente ≈ 0")
    print("                    → modelo colapsa sem aprender mais")
    print("  β = 0.1         → equilíbrio: sinal de preferência claro, KL controlada")


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 70)
    print("PASSO 3 — A Engenharia do Hiperparâmetro Beta")
    print("=" * 70)

    dpo_config = criar_dpo_config()

    print(f"\nDPOConfig instanciado:")
    print(f"  beta              = {dpo_config.beta}  ← hiperparâmetro obrigatório")
    print(f"  max_length        = {dpo_config.max_length}")
    print(f"  max_prompt_length = {dpo_config.max_prompt_length}")
    print(f"  optim             = {dpo_config.optim}")
    print(f"  learning_rate     = {dpo_config.learning_rate}")
    print(f"  num_train_epochs  = {dpo_config.num_train_epochs}")
    print(f"  fp16              = {dpo_config.fp16}")

    print(f"\nPapel matemático do beta = {BETA}:")
    print(f"  O β escala a diferença de log-ratios na função objetivo do DPO.")
    print(f"  Atua como 'imposto KL' — penaliza desvios do modelo de referência.")
    print(f"  β = {BETA} → equilíbrio entre aprender preferências e manter fluência.")

    print(f"\nAnálise do impacto de diferentes valores de beta:")
    analisar_beta()

    print(f"\n✓ DPOConfig com beta={BETA} configurado e verificado.")
    print("=" * 70)

    return dpo_config


if __name__ == "__main__":
    demo()
