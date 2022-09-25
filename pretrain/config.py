from omegaconf import DictConfig, OmegaConf
from os import path
from typing import Optional, Union

__DIR__ = path.dirname(path.realpath(__file__))


class Config:
    config: DictConfig

    def reset_config(config: Optional[Union[str, DictConfig]] = None) -> None:
        if config is None:  config = path.join(__DIR__, "config.yaml")
        Config.config = OmegaConf.load(config) if isinstance(config, str) else config

    def resolve_config(as_dict:bool = False) -> None:
        Config.config = Config.resolve_subdirs(Config.config, as_dict=as_dict)

    def resolve_subdirs(
        subconfig: Optional[Union[dict, DictConfig]] = None,
        as_dict: bool = False
    ) -> Union[dict, DictConfig]:

        exp_dir = path.relpath(path.join(config.data_dir.format(__dir__=__DIR__),
                                         config.exp_label))
        subconfig = (Config.config if subconfig is None else subconfig).copy()

        for k, v in subconfig.items():
            if k.endswith("_subdir"):
                subconfig[f"{k[:-7]}_dir"] = path.join(exp_dir, v)
                del subconfig[k]

            elif isinstance(v, DictConfig) or isinstance(v, dict):
                subconfig[k] = Config.resolve(v, as_dict=as_dict)

        return OmegaConf.to_object(subconfig) if as_dict else subconfig


Config.reset_config()
