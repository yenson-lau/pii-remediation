from omegaconf import DictConfig, OmegaConf
from os import path
from typing import Optional, Union

__DIR__ = path.dirname(path.realpath(__file__))


class Config:
    config: DictConfig

    def reset_config(config: Optional[Union[str, DictConfig]] = None) -> None:
        if config is None:  config = path.join(__DIR__, "config.yaml")
        Config.raw_config = OmegaConf.load(config) if isinstance(config, str) else config
        Config.config = Config.raw_config

    def resolve_config(as_dict:bool = False, test_override:Optional[str] = None) -> None:
        Config.config = Config.resolve(Config.raw_config,
                                       as_dict=as_dict,
                                       test_override=test_override)

    def resolve(
        subconfig: Optional[Union[dict, DictConfig]] = None,
        as_dict: bool = False,
        test_override: Optional[str] = None
    ) -> Union[dict, DictConfig]:

        subconfig = (Config.raw_config if subconfig is None else subconfig).copy()

        if (test_override is not None) and (test_override in subconfig):
            subconfig.update(subconfig[test_override])

        data_dir = path.relpath(Config.raw_config.data_dir.format(__dir__=__DIR__))

        for k, v in subconfig.items():
            if k.startswith("."):   # hidden keys
                del subconfig[k]

            elif k.endswith("_subdir"):
                subconfig[f"{k[:-7]}_dir"] = path.join(data_dir, v)
                del subconfig[k]

            elif isinstance(v, DictConfig) or isinstance(v, dict):
                subconfig[k] = Config.resolve(v, as_dict=as_dict, test_override=test_override)

        return OmegaConf.to_object(subconfig) if as_dict else subconfig


Config.reset_config()
