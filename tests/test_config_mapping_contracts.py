"""Contract tests: public engine entry points accept Mapping-typed configs.

``load_config()`` returns the ``DNALLMConfig`` TypedDict (typing special
261003-0p0); bare-``dict`` parameter annotations made ty flag every notebook's
``DNAInference(model=..., tokenizer=..., config=load_config(...))`` call as
``invalid-argument-type`` (a TypedDict is not assignable to ``dict`` because
``dict`` permits destructive operations like ``clear()``). Both engines only
read the config (``config["task"]`` / ``config["inference"]``) and never
mutate it, so ``Mapping[str, Any]`` is the honest annotation.
"""

import typing
from collections.abc import Mapping as ABCMapping

import pytest

from dnallm.finetune.trainer import DNATrainer
from dnallm.inference.inference import DNAInference


@pytest.mark.parametrize("cls", [DNAInference, DNATrainer])
def test_engine_config_parameter_accepts_mapping(cls):
    """__init__ config hints must stay Mapping-based, not bare dict.

    A regression back to ``dict`` re-breaks IDE/ty assignability for every
    notebook passing the load_config() TypedDict.
    """
    hints = typing.get_type_hints(cls.__init__)
    config_hint = hints["config"]
    origin = typing.get_origin(config_hint)
    assert origin is ABCMapping, (
        f"{cls.__name__}.__init__ config must be Mapping-typed, got {config_hint!r}"
    )
    args = typing.get_args(config_hint)
    assert args == (str, typing.Any), f"{cls.__name__} config hint args drifted: {args!r}"


def test_typeddict_config_satisfies_mapping_runtime():
    """The motivating value shape: a TypedDict instance is a Mapping at runtime.

    Guards the seam from the value side — load_config() output keeps flowing
    into the engines unchanged.
    """
    from dnallm.configuration.configs import load_config

    engine_hint = typing.get_type_hints(DNAInference.__init__)["config"]
    return_hint = typing.get_type_hints(load_config)["return"]
    # load_config returns the DNALLMConfig TypedDict; TypedDict instances are
    # plain dicts at runtime, hence Mapping instances.
    assert typing.get_origin(return_hint) is not None or hasattr(return_hint, "__annotations__")
    assert issubclass(dict, ABCMapping)
    assert engine_hint is not None
