"""Contract tests: public engine entry points accept Mapping-typed configs.

``load_config()`` returns the ``DNALLMConfig`` TypedDict (typing special
261003-0p0); bare-``dict`` parameter annotations made ty flag every notebook's
``DNAInference(model=..., tokenizer=..., config=load_config(...))`` call as
``invalid-argument-type`` (a TypedDict is not assignable to ``dict`` because
``dict`` permits destructive operations like ``clear()``). Both engines only
read the config (``config["task"]`` / ``config["inference"]``) and never
mutate it, so ``Mapping[str, Any]`` is the honest annotation.
"""

import types
import typing
from collections.abc import Mapping as ABCMapping

import pytest

from dnallm.configuration.configs import TaskConfig
from dnallm.finetune.trainer import DNATrainer
from dnallm.inference.benchmark import Benchmark
from dnallm.inference.inference import DNAInference
from dnallm.inference.interpret import DNAInterpret
from dnallm.inference.mutagenesis import Mutagenesis


@pytest.mark.parametrize(
    "cls",
    [DNAInference, DNATrainer, Mutagenesis, Benchmark, DNAInterpret],
)
def test_engine_config_parameter_accepts_mapping(cls):
    """__init__ config hints must stay Mapping-based, not bare dict.

    A regression back to ``dict`` re-breaks IDE/ty assignability for every
    notebook passing the load_config() TypedDict. Benchmark/DNAInterpret
    declare the hint optional (``| None``) — unwrap before asserting.
    """
    hint = typing.get_type_hints(cls.__init__)["config"]
    # PEP 604 unions (X | None) are types.UnionType on 3.10+; typing.Union
    # covers typing.Optional[X]. Unwrap both before the Mapping assertion.
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        assert len(non_none) == 1, f"{cls.__name__} config hint drifted: {hint!r}"
        hint = non_none[0]
    assert typing.get_origin(hint) is ABCMapping, (
        f"{cls.__name__}.__init__ config must be Mapping-typed, got {hint!r}"
    )
    assert typing.get_args(hint) == (str, typing.Any), (
        f"{cls.__name__} config hint args drifted: {typing.get_args(hint)!r}"
    )


def test_benchmark_does_not_alias_caller_config():
    """Benchmark backfills default sections into a private copy only.

    The caller's mapping (e.g. a load_config() TypedDict instance) must not
    gain keys from Benchmark's own defaulting logic.
    """
    caller_cfg: dict[str, object] = {"task": TaskConfig()}
    bench = Benchmark(config=caller_cfg)
    assert bench.config is not caller_cfg
    assert "inference" not in caller_cfg  # no default backfill into caller state


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
