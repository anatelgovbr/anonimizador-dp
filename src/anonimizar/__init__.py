"""Pacote `anonimizar`.

Ferramentas para detectar, anonimizar, treinar e avaliar modelos de
reconhecimento de entidades sensíveis em língua portuguesa.

Submódulos
----------
- SeiAnonimizar            : API principal de anonimização
- SeiAnonimizarNERTrainer  : Treinador de modelos spaCy NER
- SeiAnonimizarEvaluation  : Avaliação de desempenho de modelos

Guia rápido:
    >>> from anonimizar import SeiAnonimizar
    >>> anonymizer = SeiAnonimizar("pt_core_news_lg")
    >>> anonymizer.add_apply_patterns(['CPF', 'EMAIL'])
    >>> texto = "Meu CPF é 123.456.789-09."
    >>> entidades = anonymizer.extract_entities(texto, return_type="label_detail")
    >>> print(anonymizer.anonymize_text(texto, entidades))
    "Meu CPF é <|CPF|>."

Exemplo de Pipeline SEI-Anonimizar

1. Treinamento de um modelo NER personalizado
2. Validação do modelo em dados de avaliação
3. Anonimização de textos inéditos com o modelo recém-treinado


        >>> # 1) TREINAMENTO
        >>> from anonimizar import (
        >>>     SeiAnonimizarNERTrainer,
        >>>     SeiAnonimizarEvaluation,
        >>>     SeiAnonimizar,
        >>> )
        >>> ## Atibuindo os objetos
        >>>
        >>> trainer = SeiAnonimizarNERTrainer(output_dir='./treinamento_ner')
        >>> evaluator = SeiAnonimizarEvaluation(overlap_threshold=0.8,beta=2.0)
        >>> anonymizer = SeiAnonimizar('./treinamento_ner/')
        >>> anonymizer.add_apply_patterns(use_model_labels=True)
        >>>
        >>> ## Gerando dados
        >>>
        >>> data_list = [
        >>>     {
        >>>         'text':"Exemplo de texto com CPF 123.456.789-09.",
        >>>         'entities': [(25,39,'CPF')]},
        >>>     {
        >>>         'text':"Outro exemplo de texto com CPF 123.456.789-09.",
        >>>         'entities': [(31,45,'CPF')]},
        >>>     {
        >>>         'text':"Exemplo de texto com CPF 123.456.789-09.",
        >>>         'entities': [(25,50,'CPF')]},
        >>>     {
        >>>         "text": "João Silva, CPF 123.456.789-09, mora na Rua das Flores, 123.",
        >>>         "entities": [(16, 30, "CPF"), (40, 59, "ENDEREÇO")]
        >>>     },
        >>>     {
        >>>         "text": "Email de contato: joao@empresa.com.br, seu CPF é 123.456.789-09",
        >>>         "entities": [(18, 37, "EMAIL"), (49,63,'CPF')]
        >>>     },
        >>>     {
        >>>         "text": "Documento sem entidades para teste.",
        >>>         "entities": []
        >>>     }
        >>> ]
        >>> trainer.add_data(data_list,errors='coerce')
        >>>
        >>> ## Treino
        >>> try:
        >>>     trainer.train(
        >>>         n_iter=30,
        >>>         drop=0.2,
        >>>         batch_size=4,
        >>>         validation_split=0.2
        >>>     )
        >>>
        >>>     trainer.save_model()
        >>>
        >>>     print("Treinamento concluído com sucesso!")
        >>>     print(f"Total de exemplos de treinamento: {len(trainer.training_data)}")
        >>>
        >>> except Exception as e:
        >>>     print(f"Erro durante treinamento: {e}")
        >>>
        >>>
        >>> ## Validacao
        >>>
        >>> df_texts, df_gt = trainer.val_data_to_evaluation()
        >>>
        >>> evaluator.load_data(df_texts, df_gt)
        >>> preds = evaluator.extract_predictions(anonymizer)
        >>> results = evaluator.evaluate_model()
        >>> print(evaluator.get_summary_report())
        >>>
        >>> ## Aplicando em novos casos
        >>> entidades = anonymizer.extract_entities(
        >>>     'Oi, meu nome é Matheus, meu email é exemplo@gmail.com, e meu CPF é 123.456.789-09')
        >>> anonymizer.anonymize_text(
        >>>     'Oi, meu nome é Matheus, meu email é exemplo@gmail.com, e meu CPF é 123.456.789-09', entidades)


Fluxo completo

1. **Treinamento**
   >>> from anonimizar import SeiAnonimizarNERTrainer
   >>> trainer = SeiAnonimizarNERTrainer(labels=["CPF", "EMAIL"])
   >>> trainer.add_data(dados)
   >>> trainer.train()
   >>> trainer.save_model("./modelo_treinado")

2. **Uso do modelo**
   >>> anonymizer = SeiAnonimizar("./modelo_treinado")
   >>> anonymizer.add_apply_patterns(use_model_labels=True)

3. **Avaliação**
   >>> from anonimizar import SeiAnonimizarEvaluation
   >>> evaluator = SeiAnonimizarEvaluation(texts_path="texts.parquet",
   ...                                     ground_truth_path="gt.parquet")
   >>> evaluator.extract_predictions(anonymizer)
   >>> print(evaluator.evaluate_model())

Versão

0.10.0
"""

from .sei_anonimizar import SeiAnonimizar
from .sei_anonimizar_evaluation import SeiAnonimizarEvaluation
from .sei_anonimizar_treino import SeiAnonimizarNERTrainer

__all__ = [
    "SeiAnonimizar",
    "SeiAnonimizarEvaluation",
    "SeiAnonimizarNERTrainer",
]

__version__ = "1.0.0"
