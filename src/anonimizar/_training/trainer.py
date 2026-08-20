"""Módulo de treinamento de modelo NER.

Este módulo contém funções para treinar modelos spaCy NER
com dados no formato padronizado, incluindo curriculum learning
por fases com progressão de dificuldade.
"""

import logging
import random
import time
from typing import Any

import spacy
from spacy.training import Example

from anonimizar._common.logging import create_default_logger
from anonimizar._constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROP,
    DEFAULT_INITIAL_BASE_FRAC,
    DEFAULT_N_ITER,
    DEFAULT_VALIDATION_SPLIT,
    MIN_INIT_SAMPLE_SIZE,
)

__all__ = ["train_ner_model", "train_ner_model_curriculum"]

# Constantes para formatação de tempo
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


def _format_time(seconds: float) -> str:
    """Formata tempo em segundos para formato legível."""
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    minutes = seconds / _SECONDS_PER_MINUTE
    if minutes < _MINUTES_PER_HOUR:
        return f"{minutes:.1f}m"
    hours = minutes / _MINUTES_PER_HOUR
    return f"{hours:.1f}h"


def train_ner_model(
    nlp: Any,  # noqa: ANN401
    train_data: list[tuple[str, dict]],
    *,
    n_iter: int = DEFAULT_N_ITER,
    drop: float = DEFAULT_DROP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    initial_base_frac: float = DEFAULT_INITIAL_BASE_FRAC,
    logger: logging.Logger | None = None,
    other_pipes: list[str] | None = None,
    split_data_fn: Any = None,  # noqa: ANN401
) -> tuple[Any, dict[str, Any]]:
    """Treina o modelo NER usando os dados fornecidos.

    Args:
        nlp: Modelo spaCy carregado e configurado.
        train_data: Lista de tuplas (texto, anotações).
        n_iter: Número de iterações de treinamento. Padrão = 20.
        drop: Taxa de dropout durante atualização. Padrão = 0.35.
        batch_size: Tamanho do minilote. Padrão = 8.
        validation_split: Fração de validação delegada a ``split_data_fn``
            quando este callback é informado. O treinamento continua usando
            ``train_data`` integralmente; nenhum conjunto de validação é
            consumido ou avaliado por esta função. Padrão = 0.2.
        initial_base_frac: Fração inicial para inicializar. Padrão = 1.0.
        logger: Logger para mensagens. Se None, cria logger padrão.
        other_pipes: Pipes a desabilitar durante treinamento.
        split_data_fn: Callback opcional chamado como
            ``split_data_fn(train_ratio=1 - validation_split)`` quando
            ``validation_split > 0``. Seu retorno é ignorado; use-o somente
            para efeitos colaterais, como persistir a divisão em outro objeto.

    Raises:
        ValueError: Se train_data estiver vazio ou validation_split inválido.
        Exception: Repassa exceções do treinamento.

    Returns:
        Tupla contendo (modelo_treinado, métricas_dict):
        - Primeiro elemento: modelo spaCy treinado
        - Segundo elemento: dicionário com 'final_loss', 'iterations', 'examples_count'
    """
    if logger is None:
        logger = create_default_logger(__name__)

    if other_pipes is None:
        other_pipes = []

    try:
        if not train_data:
            msg = "Nenhum dado de treinamento disponível"
            logger.exception(msg)
            raise ValueError(msg)  # noqa: TRY301
        if not 0.0 <= validation_split < 1.0:
            msg = f"validation_split deve estar entre 0.0 e 1.0(não incluso), recebido: {validation_split}"
            logger.exception(msg)
            raise ValueError(msg)  # noqa: TRY301

        # Dividir dados se necessário
        if validation_split > 0 and split_data_fn is not None:
            split_data_fn(train_ratio=1 - validation_split)
            # Após split_data_fn, usar os dados divididos
            # Isso precisa ser obtido via callback ou modificação de estado
            final_train_data = train_data
        else:
            final_train_data = train_data

        logger.info(
            "Inicializando pipeline de treino: exemplos=%d, iterações=%d, drop=%.2f, batch=%d",
            len(final_train_data),
            n_iter,
            drop,
            batch_size,
        )
        start_time = time.time()
        iteration_times: list[float] = []

        logger.debug("Iniciando treinamento com %d exemplos por %d iterações", len(final_train_data), n_iter)
        logger.debug("Configurações: dropout=%s, batch_size=%d", drop, batch_size)

        with nlp.disable_pipes(*other_pipes):
            sample_size = int(
                min(len(final_train_data), max(MIN_INIT_SAMPLE_SIZE, int(initial_base_frac * len(final_train_data))))
            )
            examples = [
                Example.from_dict(nlp.make_doc(text), annotations)
                for text, annotations in final_train_data[:sample_size]
            ]
            nlp.initialize(lambda: examples)
            logger.info("Pipeline spaCy inicializado para treinamento.")

            for itn in range(n_iter):
                iter_start_time = time.time()
                random.shuffle(final_train_data)
                losses: dict = {}
                for batch in spacy.util.minibatch(final_train_data, size=batch_size):
                    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                    nlp.update(examples, drop=drop, losses=losses)
                iter_end_time = time.time()
                iter_duration = iter_end_time - iter_start_time
                iteration_times.append(iter_duration)

                total_elapsed = iter_end_time - start_time
                avg_time_per_itn = sum(iteration_times) / len(iteration_times)
                remaining_time = max(0.0, avg_time_per_itn * (n_iter - itn - 1))

                elapsed_str = _format_time(total_elapsed)
                estimated_str = _format_time(remaining_time)
                iter_str = _format_time(iter_duration)

                logger.debug(
                    "Iteração %d/%d: Perdas=%s | Tempo iter: %s | Decorrido: %s | Estimado restante: %s",
                    itn + 1,
                    n_iter,
                    losses,
                    iter_str,
                    elapsed_str,
                    estimated_str,
                )

        total_time = time.time() - start_time
        logger.info("Treinamento finalizado em %s.", _format_time(total_time))

        # Retornar modelo treinado e métricas
        metrics = {
            "final_loss": losses.get("ner", 0.0) if losses else 0.0,
            "iterations": n_iter,
            "examples_count": len(final_train_data),
        }
        return nlp, metrics  # noqa: TRY300

    except Exception:  # pylint: disable=broad-except
        logger.exception("Erro durante treinamento")
        raise


def _validar_fases_curriculum(phases: list[dict[str, Any]], logger: logging.Logger) -> None:
    """Valida a estrutura das fases de curriculum (train_data e epochs)."""
    for idx, fase in enumerate(phases, start=1):
        train_data = fase.get("train_data")
        if not train_data:
            msg = f"Fase {idx} não possui train_data válido"
            logger.exception(msg)
            raise ValueError(msg)
        epochs = fase.get("epochs")
        if not isinstance(epochs, int) or epochs < 1:
            msg = f"Fase {idx} deve ter epochs inteiro >= 1, recebido: {epochs}"
            logger.exception(msg)
            raise ValueError(msg)


def train_ner_model_curriculum(
    nlp: Any,  # noqa: ANN401
    phases: list[dict[str, Any]],
    *,
    drop: float = DEFAULT_DROP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_compounding: tuple[float, float, float] | None = None,
    initial_base_frac: float = DEFAULT_INITIAL_BASE_FRAC,
    logger: logging.Logger | None = None,
    other_pipes: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Treina o modelo NER seguindo um curriculum de fases.

    Cada fase define um subconjunto de dados e o número de épocas em que ele é
    apresentado, na ordem informada — permitindo progressão de dificuldade
    (janelas ``w00 → w0 → w1 → w2 → full``) e/ou de qualidade dos dados, como
    nos experimentos da estória 942.

    Args:
        nlp: Modelo spaCy carregado e configurado.
        phases: Lista de fases. Cada fase é um dicionário com:
            - ``name`` (str, opcional): nome da fase usado em logs e métricas.
            - ``train_data`` (list[tuple[str, dict]]): exemplos ``(texto,
              {"entities": [(start, end, label), ...]})`` da fase.
            - ``epochs`` (int): número de épocas da fase (>= 1).
        drop: Taxa de dropout durante atualização. Padrão = 0.35.
        batch_size: Tamanho do minilote. Padrão = 8.
        batch_compounding: Tupla ``(start, stop, rate)`` para minilotes
            crescentes via ``spacy.util.compounding`` (ex.: ``(8.0, 32.0,
            1.001)`` usado nos notebooks da 942). Se None, usa ``batch_size``
            fixo.
        initial_base_frac: Fração inicial do conjunto da 1ª fase usada em
            ``nlp.initialize()``. Padrão = 1.0.
        logger: Logger para mensagens. Se None, cria logger padrão.
        other_pipes: Pipes a desabilitar durante treinamento.

    Raises:
        ValueError: Se phases estiver vazio, alguma fase não tiver
            ``train_data`` ou tiver ``epochs`` inválido.

    Returns:
        Tupla contendo (modelo_treinado, métricas_dict):
        - Primeiro elemento: modelo spaCy treinado
        - Segundo elemento: dicionário com 'final_loss' (perda da última fase),
          'iterations' e 'total_epochs' (soma das épocas das fases),
          'examples_count' (soma dos exemplos) e 'phases' (detalhe por fase com
          'name', 'epochs', 'examples_count' e 'final_loss').
    """
    if logger is None:
        logger = create_default_logger(__name__)

    if other_pipes is None:
        other_pipes = []

    if not phases:
        msg = "Nenhuma fase de curriculum informada"
        logger.exception(msg)
        raise ValueError(msg)

    _validar_fases_curriculum(phases, logger)

    total_examples = sum(len(fase["train_data"]) for fase in phases)
    total_epochs = sum(int(fase["epochs"]) for fase in phases)

    logger.info(
        "Inicializando curriculum de treino: fases=%d, exemplos=%d, épocas=%d, drop=%.2f",
        len(phases),
        total_examples,
        total_epochs,
        drop,
    )
    start_time = time.time()
    historico_fases: list[dict[str, Any]] = []

    try:
        with nlp.disable_pipes(*other_pipes):
            first_data = phases[0]["train_data"]
            sample_size = int(min(len(first_data), max(MIN_INIT_SAMPLE_SIZE, int(initial_base_frac * len(first_data)))))
            init_examples = [
                Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in first_data[:sample_size]
            ]
            nlp.initialize(lambda: init_examples)
            logger.info("Pipeline spaCy inicializado para o curriculum.")

            compounding_size = spacy.util.compounding(*batch_compounding) if batch_compounding is not None else None

            for idx, fase in enumerate(phases, start=1):
                fase_start = time.time()
                nome = str(fase.get("name") or f"fase {idx}")
                dados = list(fase["train_data"])
                epochs = int(fase["epochs"])
                losses: dict = {}
                logger.info("Fase %d/%d (%s): %d exemplos, %d épocas", idx, len(phases), nome, len(dados), epochs)

                for _epoch in range(epochs):
                    random.shuffle(dados)
                    size = compounding_size if compounding_size is not None else batch_size
                    for batch in spacy.util.minibatch(dados, size=size):
                        examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                        nlp.update(examples, drop=drop, losses=losses)

                fase_duration = time.time() - fase_start
                fase_loss = losses.get("ner", 0.0) if losses else 0.0
                historico_fases.append(
                    {
                        "name": nome,
                        "epochs": epochs,
                        "examples_count": len(dados),
                        "final_loss": fase_loss,
                    }
                )
                logger.info(
                    "Fase %d/%d concluída em %s | perda acumulada: %.2f",
                    idx,
                    len(phases),
                    _format_time(fase_duration),
                    fase_loss,
                )

        total_time = time.time() - start_time
        logger.info("Curriculum finalizado em %s (%d épocas).", _format_time(total_time), total_epochs)

        metrics = {
            "final_loss": historico_fases[-1]["final_loss"] if historico_fases else 0.0,
            "iterations": total_epochs,
            "examples_count": total_examples,
            "total_epochs": total_epochs,
            "phases": historico_fases,
        }
        return nlp, metrics  # noqa: TRY300

    except Exception:  # pylint: disable=broad-except
        logger.exception("Erro durante treinamento por curriculum")
        raise
