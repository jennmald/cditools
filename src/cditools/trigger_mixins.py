from __future__ import annotations

import itertools
import logging
import time as ttime
from collections.abc import Sequence
from typing import Any

from ophyd import (
    Signal,
)
from ophyd.areadetector.filestore_mixins import FileStoreIterativeWrite
from ophyd.device import BlueskyInterface, Device, DeviceStatus, Staged
from ophyd.device import Component as Cpt

from .utils import ordered_dict_move_to_beginning

logger = logging.getLogger(__name__)


class TriggerBase(BlueskyInterface):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

        # If acquiring, stop.
        self.stage_sigs[self.cam.acquire] = 0  # type: ignore[attr-defined]
        self.stage_sigs[self.cam.image_mode] = "Multiple"  # type: ignore[attr-defined]
        self._acquisition_signal = self.cam.acquire  # type: ignore[attr-defined]

        self._status = None


class CDIModalSettings(Device):
    mode = Cpt(Signal, value="internal", doc="Triggering mode (internal/external)")
    scan_type = Cpt(Signal, value="step", doc="Scan type (step/fly)")
    make_directories = Cpt(Signal, value=False, doc="Make directories on the DAQ side")
    total_points = Cpt(
        Signal, value=2, doc="The total number of points to acquire overall"
    )
    triggers = Cpt(Signal, value=None, doc="Detector instances which this one triggers")


class CDIModalBase(Device):
    mode_settings = Cpt(CDIModalSettings, "")
    count_time = Cpt(
        Signal, value=1.0, doc="Exposure/count time, as specified by bluesky"
    )

    def mode_setup(self, mode: str) -> None:
        devices = [self] + [getattr(self, attr) for attr in self._sub_devices]
        attr = f"mode_{mode}"
        for dev in devices:
            if hasattr(dev, attr):
                mode_setup_method = getattr(dev, attr)
                mode_setup_method()

    def mode_internal(self) -> None:
        logger.debug("%s internal triggering %s", self.name, self.mode_settings.get())

    def mode_external(self) -> None:
        logger.debug("%s external triggering %s", self.name, self.mode_settings.get())

    @property
    def mode(self) -> Any:
        """Trigger mode (external/internal)"""
        return self.mode_settings.mode.get()

    def stage(self) -> object:
        if self._staged != Staged.yes:
            self.mode_setup(self.mode)

        return super().stage()

    def unstage(self) -> Any:
        if self.mode == "external":
            logger.info(
                "[Unstage] Stopping externally-triggered detector %s", self.name
            )
            self.stop(success=True)

        super().unstage()


class CDIModalTrigger(CDIModalBase, TriggerBase):
    def __init__(
        self,
        prefix: str,
        *args: object,
        image_name: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(prefix, *args, **kwargs)
        if image_name is None:
            image_name = "_".join([self.name, "image"])
        self._image_name = image_name
        self._external_acquire_at_stage = True

    def stop(self, success=False) -> None:
        ret = super().stop(success=success)
        self._acquisition_signal.put(0, wait=True)
        return ret

    def mode_internal(self) -> None:
        super().mode_internal()

        cam = self.cam
        cam.stage_sigs[cam.acquire] = 0
        ordered_dict_move_to_beginning(cam.stage_sigs, cam.acquire)

        cam.stage_sigs[cam.num_images] = 1
        cam.stage_sigs[cam.image_mode] = "Single"
        cam.stage_sigs[cam.trigger_mode] = "Internal"

    def mode_external(self) -> None:
        super().mode_external()
        total_points = self.mode_settings.total_points.get()

        cam = self.cam
        cam.stage_sigs[cam.num_images] = total_points
        cam.stage_sigs[cam.image_mode] = "Multiple"
        cam.stage_sigs[cam.trigger_mode] = "External"

    def stage(self) -> object:
        self._acquisition_signal.subscribe(self._acquire_changed)
        staged = super().stage()

        # In external triggering mode, the devices is only triggered once at
        # stage
        if self.mode == "external" and self._external_acquire_at_stage:
            self._acquisition_signal.put(1, wait=False)
        return staged

    def unstage(self) -> object:
        try:
            return super().unstage()
        finally:
            self._acquisition_signal.clear_sub(self._acquire_changed)

    def trigger_internal(self) -> DeviceStatus:
        if self._staged != Staged.yes:
            msg = (
                "This detector is not ready to trigger."
                "Call the stage() method before triggering."
            )
            raise RuntimeError(msg)

        self._status = DeviceStatus(self)
        self._acquisition_signal.put(1, wait=False)
        self.dispatch(self._image_name, ttime.time())
        return self._status

    def trigger_external(self) -> DeviceStatus:
        if self._staged != Staged.yes:
            msg = (
                "This detector is not ready to trigger. "
                "Call the stage() method before triggering."
            )
            raise RuntimeError(msg)

        self._status = DeviceStatus(self)
        if self._status is not None:
            self._status._finished()
        # TODO this timestamp is inaccurate!
        if self.mode_settings.scan_type.get() != "fly":
            # Don't dispatch images for fly-scans - they are bulk read at the end
            self.dispatch(self._image_name, ttime.time())
        return self._status

    def trigger(self):
        mode_trigger = getattr(self, f"trigger_{self.mode}")
        return mode_trigger()

    def _acquire_changed(self, value=None, old_value=None) -> None:
        """This is called when the 'acquire' signal changes."""
        if self._status is None:
            return
        if (old_value == 1) and (value == 0):
            # Negative-going edge means an acquisition just finished.
            self._status._finished()


class FileStoreBulkReadable(FileStoreIterativeWrite):
    def _reset_data(self) -> None:
        self._datum_uids.clear()
        self._point_counter = itertools.count()

    def bulk_read(self, timestamps: Sequence[float]) -> dict[str, list[str]]:
        image_name = self.image_name

        uids = [self.generate_datum(self.image_name, ts, {}) for ts in timestamps]

        # clear so unstage will not save the images twice:
        self._reset_data()
        return {image_name: uids}

    @property
    def image_name(self) -> str:
        return self.parent._image_name  # type: ignore[attr-defined]
