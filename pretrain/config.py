from omegaconf import DictConfig, OmegaConf
from os import path
from typing import Optional, Union

__DIR__ = path.dirname(path.realpath(__file__))


class Config:
    config: DictConfig

    def set_config(config: Union[str, DictConfig]):
        Config.config = OmegaConf.load(config) if isinstance(config, str) else config

    def get_exp_dir():
        config = Config.config
        return path.relpath(path.join(config.data_dir.format(__dir__=__DIR__),
                                      config.exp_label))

    def resolve(
        subconfig: Optional[Union[dict, DictConfig]] = None,
        as_dict: bool = False
    ) -> Union[dict, DictConfig]:

        exp_dir = Config.get_exp_dir()
        subconfig = Config.config.copy() if subconfig is None else subconfig.copy()

        for k, v in subconfig.items():
            if k.endswith("_subdir"):
                subconfig[f"{k[:-7]}_dir"] = path.join(exp_dir, v)
                del subconfig[k]

            elif isinstance(v, DictConfig) or isinstance(v, dict):
                subconfig[k] = Config.resolve(v, as_dict=as_dict)

        return OmegaConf.to_object(subconfig) if as_dict else subconfig

Config.set_config(path.join(__DIR__, "config.yaml"))
