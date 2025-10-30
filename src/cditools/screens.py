from __future__ import annotations

from typing import Optional, Union

from ophyd import (
    CamBase,
    ImagePlugin,
    ProsilicaDetector,
    ProsilicaDetectorCam,
    ROIPlugin,
    ROIStatPlugin,
    StatsPluginV33,
    TransformPlugin,
)
from ophyd import Component as Cpt
from ophyd.areadetector.plugins import PluginBase


class ProsilicaCamBase(ProsilicaDetector):
    cam = Cpt(ProsilicaDetectorCam, "cam1:")  # VMB1????
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    roistat1 = Cpt(ROIStatPlugin, "ROISTAT1:")
    _default_plugin_graph: Optional[dict[PluginBase, Union[CamBase, PluginBase]]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roistat1.kind = "hinted"
        self._use_default_plugin_graph: bool = True

    @property
    def default_plugin_graph(
        self,
    ) -> Optional[dict[PluginBase, Union[CamBase, PluginBase]]]:
        return self._default_plugin_graph

    def _stage_plugin_graph(
        self, plugin_graph: dict[PluginBase, Union[CamBase, PluginBase]]
    ):
        for target, source in plugin_graph.items():
            self.stage_sigs[target.nd_array_port] = source.port_name.get()
            self.stage_sigs[target.enable] = True

    def stage(self):
        if self._use_default_plugin_graph and self.default_plugin_graph is not None:
            self._stage_plugin_graph(self.default_plugin_graph)

        return super().stage()


class StandardProsilicaCam(ProsilicaCamBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.cam.wait_for_plugins] = "No"
        self._default_plugin_graph = {
            self.cam: self.roistat1
        }  # ask tom if this should be reversed

    def stage(self):
        return super().stage()
