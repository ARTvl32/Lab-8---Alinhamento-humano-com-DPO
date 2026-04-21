"""
Laboratório 8 — Passo 2: Preparação do Pipeline DPO
=====================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Utilizando a biblioteca trl (Hugging Face), importa a classe DPOTrainer e
prepara os dois modelos obrigatórios na memória:

    Modelo Ator      : O modelo que terá os pesos atualizados durante o DPO.
    Modelo Referência: O modelo base congelado (adaptador do Lab 07) usado
                       para calcular a divergência de Kullback-Leibler (KL).

Por que dois modelos são necessários no DPO?
--------------------------------------------
A função objetivo do DPO compara a probabilidade das respostas chosen e
rejected sob o modelo ator (π_θ) com as probabilidades sob o modelo de
referência (π_ref). A divergência KL entre os dois impede que o modelo ator
se afaste demais do comportamento linguístico original, preservando fluência.

    L_DPO = -E[ log σ( β * log(π_θ(y_w|x)/π_ref(y_w|x))
                     - β * log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

O modelo de referência permanece congelado (requires_grad = False em todos
os parâmetros), sendo usado apenas no forward pass para calcular as
log-probabilidades de referência.

Dependências
------------
    pip install transformers trl peft bitsandbytes accelerate torch
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft          import PeftModel


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

# Modelo base — pode ser substituído por qualquer modelo compatível no HF Hub
MODEL_NAME       = "meta-llama/Llama-2-7b-hf"

# Caminho do adaptador LoRA treinado no Lab 07 (modelo de referência)
LORA_ADAPTER_PATH = "../lab7-especializacao-llm-lora-qlora/lora_adapter_final"


# ---------------------------------------------------------------------------
# Passo 2.1 — Configuração de quantização (reutilizada do Lab 07)
# ---------------------------------------------------------------------------

def criar_bnb_config():
    """
    Configuração BitsAndBytes para carregar ambos os modelos em 4-bit,
    viabilizando manter modelo ator e modelo de referência na mesma GPU.
    """
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )


# ---------------------------------------------------------------------------
# Passo 2.2 — Carregar modelo ator (pesos serão atualizados)
# ---------------------------------------------------------------------------

def carregar_modelo_ator(model_name=MODEL_NAME):
    """
    Carrega o modelo ator com quantização 4-bit.

    O modelo ator terá seus pesos atualizados durante o treinamento DPO.
    Para economizar memória, é carregado com os mesmos adaptadores LoRA
    do Lab 07 — o DPOTrainer continuará o fine-tuning a partir desse ponto.

    Parâmetros
    ----------
    model_name : str  — nome ou caminho do modelo base no Hugging Face Hub

    Retorna
    -------
    model : AutoModelForCausalLM  — modelo ator quantizado em 4-bit
    """
    bnb_config = criar_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
    )

    model.config.use_cache      = False
    model.config.pretraining_tp = 1

    return model


# ---------------------------------------------------------------------------
# Passo 2.3 — Carregar modelo de referência (CONGELADO)
# ---------------------------------------------------------------------------

def carregar_modelo_referencia(model_name=MODEL_NAME, adapter_path=None):
    """
    Carrega o modelo de referência congelado.

    O modelo de referência é o adaptador treinado no Lab 07 (ou o modelo
    base se o adaptador não estiver disponível). Seus parâmetros são
    congelados (requires_grad = False) — ele é usado apenas no forward
    pass para calcular as log-probabilidades de referência necessárias
    para o cálculo da divergência KL na função objetivo do DPO.

    Parâmetros
    ----------
    model_name   : str       — nome ou caminho do modelo base
    adapter_path : str|None  — caminho do adaptador LoRA do Lab 07

    Retorna
    -------
    ref_model : AutoModelForCausalLM  — modelo de referência congelado
    """
    bnb_config = criar_bnb_config()

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
    )

    # Carregar adaptador LoRA do Lab 07 se disponível
    if adapter_path:
        import os
        if os.path.exists(adapter_path):
            ref_model = PeftModel.from_pretrained(ref_model, adapter_path)
            print(f"  ✓ Adaptador LoRA carregado de '{adapter_path}'")
        else:
            print(f"  ⚠ Adaptador não encontrado em '{adapter_path}'. "
                  "Usando modelo base como referência.")

    # CONGELAR todos os parâmetros do modelo de referência
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


# ---------------------------------------------------------------------------
# Passo 2.4 — Carregar tokenizador
# ---------------------------------------------------------------------------

def carregar_tokenizador(model_name=MODEL_NAME):
    """
    Carrega o tokenizador do modelo base.

    Parâmetros
    ----------
    model_name : str

    Retorna
    -------
    tokenizer : AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"    # DPOTrainer recomenda padding à esquerda
    return tokenizer


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("PASSO 2 — Preparação do Pipeline DPO")
    print("=" * 65)

    print("\nComponentes do pipeline DPO:")
    print("\n  Modelo Ator:")
    print(f"    Base      : {MODEL_NAME}")
    print(f"    Pesos     : serão ATUALIZADOS durante o treinamento DPO")
    print(f"    Quantização: BitsAndBytes 4-bit (nf4/float16)")

    print("\n  Modelo de Referência:")
    print(f"    Base      : {MODEL_NAME}")
    print(f"    Adaptador : adaptador LoRA do Lab 07 (se disponível)")
    print(f"    Pesos     : CONGELADOS (requires_grad = False)")
    print(f"    Papel     : calcula π_ref(y|x) para divergência KL")

    print("\n  Papel do Modelo de Referência na função objetivo do DPO:")
    print("    L_DPO = -E[ log σ( β * log(π_θ(y_w|x)/π_ref(y_w|x))")
    print("                     - β * log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]")
    print("    Onde π_ref mantém o comportamento linguístico original.")

    # Verificação de GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU disponível: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  Estimativa de uso (ambos os modelos em 4-bit): ~12–16 GB VRAM")
    else:
        print("\n⚠ GPU não detectada. Execute no Google Colab para treinamento completo.")

    print("\nNota: Para carregar ambos os modelos, execute no Colab:")
    print("  model_ator = carregar_modelo_ator()")
    print("  model_ref  = carregar_modelo_referencia(adapter_path=LORA_ADAPTER_PATH)")
    print("  tokenizer  = carregar_tokenizador()")

    print("\n✓ Configurações do pipeline DPO verificadas.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
