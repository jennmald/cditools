from __future__ import annotations

from typing import Optional

from ophyd import (
    CamBase,
    ImagePlugin,
    ProsilicaDetector,
    ProsilicaDetectorCam,
    ROIPlugin,
)
from ophyd import Component as Cpt
from ophyd.areadetector.plugins import (
    PluginBase,
    ROIStatPlugin,
    StatsPlugin,
    TransformPlugin,
)


class ProsilicaCamBase(ProsilicaDetector):
    cam = Cpt(ProsilicaDetectorCam, "cam1:")  # VMB1????
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPlugin, "Stats1:")
    stats2 = Cpt(StatsPlugin, "Stats2:")
    stats3 = Cpt(StatsPlugin, "Stats3:")
    stats4 = Cpt(StatsPlugin, "Stats4:")
    stats5 = Cpt(StatsPlugin, "Stats5:")
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    roistat1 = Cpt(ROIStatPlugin, "ROISTAT1:")
    _default_plugin_graph: Optional[dict[PluginBase, CamBase | PluginBase]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roistat1.kind = "hinted"
        self._use_default_plugin_graph: bool = True

    @property
    def default_plugin_graph(
        self,
    ) -> Optional[dict[PluginBase, CamBase | PluginBase]]:
        return self._default_plugin_graph

    def _stage_plugin_graph(self, plugin_graph: dict[PluginBase, CamBase | PluginBase]):
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

    # def insert_screen(self, state: float):
    #     if state > 0.0:  # should this be != to zero?
    #         self.fs.y.set(0.0)  # insert screen

    # def remove_screen(self, state: float):
    #     if (
    #         state == 0.0
    #     ):  # 25 mm is the current out value, could change after operations
    #         self.fs.y.set(25.0)  # remove screen

    # possible wrapper for after scan cleanup
    # def set_screen(self, state: float):
    #     # the screen is "in" (in the beam) when state is 1.0
    #     if state == 0.0:
    #         pass
    #     # otherwise screen is "out" (out of the beam), for normal operations
    #     elif state > 0.0:
    #         pass
    #     else:
    #         raise ValueError(f"Invalid state {state} for VPM screen.")
    #     #return super().set(state)


cam_A1 = StandardProsilicaCam("XF:09IDA-OP:1{DM-Cam:1}", name="cam_A1")
cam_A2 = StandardProsilicaCam("XF:09IDA-BI:1{WBStop-Cam:2}", name="cam_A2")
cam_A3 = StandardProsilicaCam("XF:09IDB-BI:1{VPM-Cam:3}", name="cam_A3")
cam_A4 = StandardProsilicaCam("XF:09IDB-BI:1{HPM-Cam:4}", name="cam_A4")
