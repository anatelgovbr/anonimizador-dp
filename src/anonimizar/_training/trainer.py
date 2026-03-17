"""Módulo de treinamento de modelo NER.

Este módulo contém funções para treinar modelos spaCy NER
com dados no formato padronizado.
"""

import logging
import random
import time
from typing import Any

import spacy
from spacy.training import Example

from anonimizar._constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROP,
    DEFAULT_INITIAL_BASE_FRAC,
    DEFAULT_N_ITER,
    DEFAULT_VALIDATION_SPLIT,
    MIN_INIT_SAMPLE_SIZE,
)

__all__ = ["train_ner_model"]

# Constantes para formatação de tempo
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


def train_ner_model(  # noqa: C901, PLR0915
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
        validation_split: Fração para validação (0-1). Padrão = 0.2.
        initial_base_frac: Fração inicial para inicializar. Padrão = 1.0.
        logger: Logger para mensagens. Se None, cria logger padrão.
        other_pipes: Pipes a desabilitar durante treinamento.
        split_data_fn: Função para dividir dados se validation_split > 0.

    Raises:
        ValueError: Se train_data estiver vazio ou validation_split inválido.
        Exception: Repassa exceções do treinamento.

    Returns:
        Tupla contendo (modelo_treinado, métricas_dict):
        - Primeiro elemento: modelo spaCy treinado
        - Segundo elemento: dicionário com 'final_loss', 'iterations', 'examples_count'
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if other_pipes is None:
        other_pipes = []

    def _format_time(seconds: float) -> str:
        """Formata tempo em segundos para formato legível."""
        if seconds < _SECONDS_PER_MINUTE:
            return f"{seconds:.1f}s"
        minutes = seconds / _SECONDS_PER_MINUTE
        if minutes < _MINUTES_PER_HOUR:
            return f"{minutes:.1f}m"
        hours = minutes / _MINUTES_PER_HOUR
        return f"{hours:.1f}h"

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
