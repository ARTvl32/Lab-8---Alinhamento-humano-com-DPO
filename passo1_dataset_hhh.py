"""
Laboratório 8 — Passo 1: Construção do Dataset de Preferências (The HHH Dataset)
==================================================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
O DPO não usa dados de instrução simples. Ele exige pares de preferência.
Este script constrói e valida o dataset no formato .jsonl contendo as
3 chaves obrigatórias por linha:

    prompt   : A instrução ou pergunta
    chosen   : A resposta segura e alinhada (HHH)
    rejected : A resposta prejudicial ou inadequada

O dataset contém pelo menos 30 exemplos focados em restrições de segurança
e adequação de tom corporativo.

Nota sobre IA
-------------
O brainstorming dos pares chosen/rejected foi gerado/complementado com IA,
com revisão crítica e curadoria manual de todos os 30 pares por Arthur.
"""

import json
import os


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

DATASET_FILE = "data/dataset_hhh.jsonl"
MIN_EXAMPLES = 30
REQUIRED_KEYS = {"prompt", "chosen", "rejected"}


# ---------------------------------------------------------------------------
# Passo 1 — Carregar e validar o dataset
# ---------------------------------------------------------------------------

def carregar_dataset(caminho=DATASET_FILE):
    """
    Carrega o dataset HHH do arquivo .jsonl.

    Cada linha deve conter estritamente as 3 chaves obrigatórias:
    prompt, chosen e rejected.

    Parâmetros
    ----------
    caminho : str  — caminho para o arquivo .jsonl

    Retorna
    -------
    exemplos : list[dict]  — lista de pares de preferência
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Dataset não encontrado em '{caminho}'. "
            "Execute o script a partir da raiz do repositório."
        )

    exemplos = []
    with open(caminho, "r", encoding="utf-8") as f:
        for i, linha in enumerate(f, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                par = json.loads(linha)
                exemplos.append(par)
            except json.JSONDecodeError as e:
                raise ValueError(f"Erro de JSON na linha {i}: {e}")

    return exemplos


def validar_dataset(exemplos):
    """
    Valida que o dataset possui:
        1. Pelo menos MIN_EXAMPLES exemplos
        2. Estritamente as chaves: prompt, chosen, rejected em cada linha
        3. Nenhum valor vazio

    Parâmetros
    ----------
    exemplos : list[dict]

    Retorna
    -------
    bool  — True se válido, levanta AssertionError caso contrário
    """
    # Validação 1: quantidade mínima
    assert len(exemplos) >= MIN_EXAMPLES, (
        f"Dataset deve ter pelo menos {MIN_EXAMPLES} exemplos. "
        f"Encontrados: {len(exemplos)}"
    )

    # Validação 2: chaves obrigatórias e valores não-vazios
    for i, par in enumerate(exemplos, start=1):
        chaves_presentes = set(par.keys())

        assert chaves_presentes == REQUIRED_KEYS, (
            f"Exemplo {i}: chaves incorretas. "
            f"Esperado: {REQUIRED_KEYS}, encontrado: {chaves_presentes}"
        )

        for chave in REQUIRED_KEYS:
            assert par[chave].strip(), (
                f"Exemplo {i}: valor vazio para a chave '{chave}'"
            )

    return True


def carregar_como_hf_dataset(caminho=DATASET_FILE):
    """
    Carrega o dataset .jsonl no formato Hugging Face Dataset,
    pronto para ser passado ao DPOTrainer.

    Parâmetros
    ----------
    caminho : str

    Retorna
    -------
    dataset : datasets.Dataset
    """
    from datasets import load_dataset
    dataset = load_dataset("json", data_files={"train": caminho}, split="train")
    return dataset


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("PASSO 1 — Construção do Dataset de Preferências (HHH Dataset)")
    print("=" * 65)

    print(f"\nCarregando dataset de '{DATASET_FILE}'...")
    exemplos = carregar_dataset()

    print(f"✓ {len(exemplos)} exemplos carregados.")

    print(f"\nValidando estrutura do dataset...")
    validar_dataset(exemplos)
    print(f"✓ Todas as {len(exemplos)} linhas possuem estritamente as chaves:")
    print(f"  {REQUIRED_KEYS}")
    print(f"✓ Mínimo de {MIN_EXAMPLES} exemplos: APROVADO")

    print(f"\nExemplos do dataset (primeiros 3):")
    for i, par in enumerate(exemplos[:3], start=1):
        print(f"\n  [{i}]")
        print(f"  prompt   : {par['prompt']}")
        print(f"  chosen   : {par['chosen'][:80]}...")
        print(f"  rejected : {par['rejected'][:80]}...")

    # Estatísticas
    lens_chosen   = [len(p["chosen"].split())   for p in exemplos]
    lens_rejected = [len(p["rejected"].split()) for p in exemplos]

    print(f"\nEstatísticas do dataset:")
    print(f"  Total de exemplos            : {len(exemplos)}")
    print(f"  Comprimento médio chosen     : {sum(lens_chosen)/len(lens_chosen):.1f} palavras")
    print(f"  Comprimento médio rejected   : {sum(lens_rejected)/len(lens_rejected):.1f} palavras")

    print(f"\n✓ Dataset HHH validado e pronto para o DPOTrainer.")
    print("=" * 65)

    return exemplos


if __name__ == "__main__":
    demo()
