"""Registry de validators contextuais por tipo de entidade.

Este módulo fornece a infraestrutura para registrar e acessar funções de
validação contextual para diferentes tipos de entidades sensíveis.
"""

from collections.abc import Callable

# Tipo do validator: recebe (text, start, end, entity_text, **kwargs) -> bool | str
ContextValidator = Callable[..., bool | str]

# Registry populado pela task 006 (context.py)
VALIDATOR_REGISTRY: dict[str, ContextValidator] = {}


def register_validator(label: str) -> Callable[[ContextValidator], ContextValidator]:
    """Decorator para registrar um validator no registry.

    Args:
        label (str): Label da entidade (ex: 'CPF', 'RG', 'EMAIL')

    Returns:
        Callable: Decorator que registra a função no VALIDATOR_REGISTRY

    Example:
        >>> @register_validator("CPF")
        >>> def validate_cpf_context(text, start, end, entity_text, **kwargs):
        ...     return True
    """

    def decorator(fn: ContextValidator) -> ContextValidator:
        VALIDATOR_REGISTRY[label] = fn
        return fn

    return decorator


def get_validator(label: str) -> ContextValidator | None:
    """Retorna o validator para um label, ou None se não registrado.

    Args:
        label (str): Label da entidade

    Returns:
        ContextValidator | None: Função de validação ou None se não encontrada
    """
    return VALIDATOR_REGISTRY.get(label)
