from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from ranzen import implements

__all__ = [
    "LocalSearchPathPlugin",
]


class LocalSearchPathPlugin(SearchPathPlugin):
    @implements(SearchPathPlugin)
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(provider="diffusion-app", path="file://static_confs")
