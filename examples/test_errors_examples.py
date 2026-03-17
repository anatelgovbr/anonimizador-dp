"""
Exemplos de uso das políticas de erros no SeiAnonimizarNERTrainer.

Este arquivo demonstra os comportamentos de:
- errors: 'raise', 'coerce', 'ignore'
- auto_clean: True/False
- strict_clean: True/False
- resolve_conflicts: 'raise', 'coerce', 'ignore'

Execute com:
    python examples/test_errors_examples.py
"""

from anonimizar.sei_anonimizar_treino import SeiAnonimizarNERTrainer


def print_separator(title: str) -> None:
    """Imprime um separador formatado."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def main():
    """Executa todos os exemplos de políticas de erros."""
    print("Exemplos de Políticas de Erros no SeiAnonimizarNERTrainer")
    print("=" * 60)

    # =========================================================================
    # 1. Entidades corretas vs inválidas
    # =========================================================================
    print_separator("1. Entidades Corretas vs Inválidas")

    trainer = SeiAnonimizarNERTrainer(labels=["CPF", "PLACA"])

    # Entidade correta
    text = "CPF: 123.456.789-09"
    entities = [(5, 19, "CPF")]  # Offset correto
    result = trainer.validate_entities(text, entities)
    print(f"Texto: {text}")
    print(f"Entities: {entities}")
    print(f"validate_entities: {result} (esperado: True)")

    # Entidade com espaços extras
    text_spaces = "CPF: 123.456.789-09  "
    entities_spaces = [(5, 21, "CPF")]  # Inclui espaços
    result_spaces = trainer.validate_entities(text_spaces, entities_spaces)
    print(f"\nTexto: '{text_spaces}'")
    print(f"Entities: {entities_spaces}")
    print(f"validate_entities: {result_spaces} (esperado: False)")

    # Entidade com offset desalinhado (BILUO)
    text_biluo = "EMAIL: teste@exemplo.com"
    entities_biluo = [(14, 21, "EMAIL")]  # Não alinhado ao token
    result_biluo = trainer.validate_entities(text_biluo, entities_biluo)
    print(f"\nTexto: '{text_biluo}'")
    print(f"Entities: {entities_biluo}")
    print(f"validate_entities: {result_biluo} (esperado: False)")

    # =========================================================================
    # 2. Diferenças entre errors='ignore' vs errors='coerce'
    # =========================================================================
    print_separator("2. errors='ignore' vs errors='coerce'")

    # errors='ignore' - mantém entidades como estão (sem correção)
    print("\n2.1 errors='ignore' + auto_clean=True")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "CPF: 123.456.789-09  ", "entities": [(5, 21, "CPF")]}]
    trainer.add_data(data, errors="ignore", auto_clean=True)
    print(f"Input entities: [(5, 21, 'CPF')]  (com espaços)")
    print(f"Stored entities: {trainer.training_data[0][1]['entities']}")
    print(f"Comportamento: Mantém entidades como estão")

    # errors='coerce' - corrige espaços
    print("\n2.2 errors='coerce' + auto_clean=True")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "CPF: 123.456.789-09  ", "entities": [(5, 21, "CPF")]}]
    trainer.add_data(data, errors="coerce", auto_clean=True)
    print(f"Input entities: [(5, 21, 'CPF')]  (com espaços)")
    print(f"Stored entities: {trainer.training_data[0][1]['entities']}")
    print(f"Comportamento: Corrige espaços extras")

    # errors='ignore' - mantém entidades inválidas
    print("\n2.3 errors='ignore' + auto_clean=False")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "CPF: 123.456.789-00", "entities": [(20, 30, "CPF")]}]
    trainer.add_data(data, errors="ignore", auto_clean=False)
    print(f"Input entities: [(20, 30, 'CPF')]  (offset inválido)")
    print(f"Training data length: {len(trainer.training_data)}")
    print(f"Comportamento: Mantém entidades inválidas")

    # =========================================================================
    # 3. Impacto de auto_clean=True/False
    # =========================================================================
    print_separator("3. Impacto de auto_clean=True/False")

    # auto_clean=True - aplica limpeza automática
    print("\n3.1 auto_clean=True (padrão)")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "  CPF: 123.456.789-09  ", "entities": [(1, 21, "CPF")]}]
    trainer.add_data(data, auto_clean=True)
    print(f"Input: '  CPF: 123.456.789-09  ' com [(1, 21, 'CPF')]")
    print(f"Stored: {trainer.training_data[0][1]['entities']}")
    print(f"Comportamento: Remove espaços, valida BILUO")

    # auto_clean=False - sem limpeza automática
    print("\n3.2 auto_clean=False")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "  CPF: 123.456.789-09  ", "entities": [(1, 21, "CPF")]}]
    trainer.add_data(data, auto_clean=False)
    print(f"Input: '  CPF: 123.456.789-09  ' com [(1, 21, 'CPF')]")
    print(f"Stored: {trainer.training_data[0][1]['entities']}")
    print(f"Comportamento: Não aplica limpeza, apenas valida")

    # =========================================================================
    # 4. Debug de entidades
    # =========================================================================
    print_separator("4. Debug de Entidades")

    trainer = SeiAnonimizarNERTrainer(labels=["CPF", "PLACA"])
    text = "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007."
    entities = [(25, 41, "CPF"), (71, 80, "PLACA")]

    print(f"Texto: {text}")
    print(f"Entities: {entities}")
    print("\nSaída do debug_entities:")
    trainer.debug_entities(text, entities)

    # =========================================================================
    # 5. Cenários com conflitos de entidades
    # =========================================================================
    print_separator("5. Conflitos de Entidades")

    # Dados com duplicatas
    print("\n5.1 Duplicatas")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "CPF: 123.456.789-09", "entities": [(5, 19, "CPF"), (5, 19, "CPF")]}]
    trainer.add_data(data, resolve_conflicts="coerce")
    print(f"Input: 2x [(5, 19, 'CPF')]")
    print(f"Stored: {trainer.training_data[0][1]['entities']}")
    print(f"Comportamento: Remove duplicata automaticamente")

    # Dados com sobreposição
    print("\n5.2 Sobreposição")
    trainer = SeiAnonimizarNERTrainer(labels=["CPF"])
    data = [{"text": "CPF: 123.456.789-09", "entities": [(5, 19, "CPF"), (10, 20, "CPF")]}]
    trainer.add_data(data, resolve_conflicts="coerce")
    print(f"Input: [(5, 19, 'CPF'), (10, 20, 'CPF')]")
    print(f"Stored: {trainer.training_data[0][1]['entities']}")
    print(f"Comportamento: Resolve sobreposição automaticamente")

    # =========================================================================
    # 6. Tabela resumo
    # =========================================================================
    print_separator("6. Tabela Resumo")

    print("""
| errors  | auto_clean | Comportamento                                    |
|---------|------------|-------------------------------------------------|
| 'raise' | False      | Levanta erro em entidades inválidas            |
| 'raise' | True       | Corrige espaços, levanta erro se persistir      |
| 'coerce'| False      | Descarta entidades inválidas                   |
| 'coerce'| True       | Corrige espaços, descarta BILUO inválido        |
| 'ignore'| False      | Mantém entidades como estão                     |
| 'ignore'| True       | Mantém entidades como estão (sem correção)      |
""")

    print("\nExecução concluída com sucesso!")


if __name__ == "__main__":
    main()
