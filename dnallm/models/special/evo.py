import os
from glob import glob
from ...utils import is_flash_attention_capable, is_fp8_capable


evo2_models = {
    "evo2_1b_base": "evo2-1b-8k",
    "evo2_7b_base": "evo2-7b-8k",
    "evo2_7b_262k": "evo2-7b-262k",
    "evo2_7b": "evo2-7b-1m",
    "evo2_40b_base": "evo2-40b-8k",
    "evo2_40b": "evo2-40b-1m",
    "evo2_7b_microviridae": "evo2-7b-8k",
}

evo_models = {
    "evo-1.5-8k-base": "evo-1-131k-base",
    "evo-1-8k-base": "evo-1-8k-base",
    "evo-1-131k-base": "evo-1-131k-base",
    "evo-1-8k-crispr": "evo-1-8k-base",
    "evo-1-8k-transposon": "evo-1-8k-base",
}


def _handle_evo2_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO2 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO2 model, None otherwise
    """

    for m in evo2_models:
        if m in model_name.lower():
            try:
                from evo2 import Evo2  # pyright: ignore[reportMissingImports]
                from vortex.model.tokenizer import CharLevelTokenizer

                # Overwrite Evo2 to avoid init errors
                class CustomEvo2(Evo2):
                    def __init__(self):
                        pass

                    load_evo2_model = Evo2.load_evo2_model

            except ImportError as e:
                raise ImportError(
                    f"EVO2 package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/ArcInstitute/evo2"
                ) from e

            model_path = (
                glob(model_name + "/*.pt")[0]
                if os.path.isdir(model_name)
                else model_name
            )
            # Check the dependencies and find the correct config file
            is_fp8 = is_fp8_capable()
            has_flash_attention = is_flash_attention_capable()
            config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            if is_fp8:
                suffix2 = ""
            else:
                suffix2 = "-noFP8"
            suffix = suffix1 + suffix2 + ".yml"
            config_path = os.path.join(config_dir, evo2_models[m] + suffix)
            # Load the model with the built-in method
            evo2_model = CustomEvo2()
            if source.lower() == "local":
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=model_path,
                    config_path=config_path,
                )
            else:
                from ..model import _get_model_path_and_imports

                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                downloaded_model_path = os.path.join(
                    downloaded_model_path, m + ".pt"
                )
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=downloaded_model_path,
                    config_path=config_path,
                )
            tokenizer = CharLevelTokenizer(512)
            evo2_model.tokenizer = tokenizer
            evo2_model._model_path = downloaded_model_path
            return evo2_model, tokenizer

    return None


def _handle_evo1_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO1 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO1 model, None otherwise
    """

    for m in evo_models:
        if m in model_name.lower():
            try:
                import yaml
                from evo import Evo  # pyright: ignore[reportMissingImports]
                from stripedhyena.utils import dotdict
                from stripedhyena.model import StripedHyena
                from stripedhyena.tokenizer import CharLevelTokenizer

                # Overwrite Evo2 to avoid init errors
                class CustomEvo1(Evo):
                    def __init__(self):
                        self.device = None
                        self.model = None

                    def load_checkpoint(
                        self,
                        model_name: str = "evo-1-8k-base",
                        revision: str = "main",
                        config_path: str | None = None,
                        modules: dict | None = None,
                    ) -> StripedHyena:
                        autoconfig = modules["AutoConfig"]
                        automodelforcausallm = modules["AutoModelForCausalLM"]
                        # Load the model configuration and weights
                        model_config = autoconfig.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            revision=revision,
                        )
                        model_config.use_cache = True
                        model = automodelforcausallm.from_pretrained(
                            model_name,
                            config=model_config,
                            trust_remote_code=True,
                            revision=revision,
                        )
                        state_dict = model.backbone.state_dict()
                        del model
                        del model_config
                        global_config = dotdict(
                            yaml.safe_load(open(config_path))
                        )
                        model = StripedHyena(global_config)
                        model.load_state_dict(state_dict, strict=True)
                        model.to_bfloat16_except_poles_residues()
                        return model

            except ImportError as e:
                raise ImportError(
                    f"EVO-1 package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/evo-design/evo"
                ) from e

            has_flash_attention = is_flash_attention_capable()
            config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            suffix = suffix1 + ".yml"
            config_path = os.path.join(config_dir, evo_models[m] + suffix)
            # Load the model with the built-in method
            from ..model import _get_model_path_and_imports

            evo_model = CustomEvo1()
            revision = (
                "1.1_fix"
                if "." in model_name and source == "huggingface"
                else "main"
            )
            _, modules = _get_model_path_and_imports(
                model_name, source, revision=revision
            )
            evo_model.model = evo_model.load_checkpoint(
                model_name=model_name,
                revision=revision,
                config_path=config_path,
                modules=modules,
            )
            tokenizer = CharLevelTokenizer(512)
            evo_model.tokenizer = tokenizer
            return evo_model, tokenizer

    return None
