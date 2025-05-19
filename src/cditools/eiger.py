from __future__ import annotations

import datetime
import logging
import os
import time as ttime
from collections import OrderedDict, deque
from pathlib import PurePath
from types import SimpleNamespace

import h5py
from ophyd import Component as Cpt
from ophyd import (
    Device,
    EpicsPathSignal,
    EpicsSignal,
    ImagePlugin,
    Signal,
    SingleTrigger,
)
from ophyd.areadetector import EigerDetector
from ophyd.areadetector.base import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.filestore_mixins import FileStoreBase  # , new_short_uid


logger = logging.getLogger(__name__)

DEFAULT_DATUM_DICT = {"data": None, "omega": None}

# TODO: convert it to Enum class.
INTERNAL_SERIES = 0
INTERNAL_ENABLE = 1
EXTERNAL_SERIES = 2
EXTERNAL_ENABLE = 3


class EigerDetectorWithPlugins(EigerDetector):
    hdf5 = ...

class EigerSingleTrigger(SingleTrigger, EigerDetectorWithPlugins):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove `cam.acquire` since we want to keep the camera acquiring
        self.stage_sigs.pop("cam.acquire")
        self.stage_sigs.update(
            {"cam.compression_algo": "BS LZ4"}
        )

    def collect_asset_docs(self):
        asset_docs_cache = []

        # Get the Resource which was produced when the detector was staged.
        ((name, resource),) = self.file.collect_asset_docs()

        asset_docs_cache.append(("resource", resource))
        self._datum_ids = DEFAULT_DATUM_DICT
        # Generate Datum documents from scratch here, because the detector was
        # triggered externally by the DeltaTau, never by ophyd.
        resource_uid = resource["uid"]
        # We are currently generating only one datum document for all frames, that's why
        #   we use the 0th index below.
        #
        # Uncomment & update the line below if more datum documents are needed:
        # for i in range(num_points):

        seq_id = self.cam.sequence_id.get()

        self._master_file = (
            f"{resource['root']}/{resource['resource_path']}_{seq_id}_master.h5"
        )
        if not os.path.isfile(self._master_file):
            raise RuntimeError(f"File {self._master_file} does not exist")

        # The pseudocode below is from Tom Caswell explaining the relationship between resource, datum, and events.
        #
        # resource = {
        #     "resource_id": "RES",
        #     "resource_kwargs": {},  # this goes to __init__
        #     "spec": "AD-EIGER-MX",
        #     ...: ...,
        # }
        # datum = {
        #     "datum_id": "a",
        #     "datum_kwargs": {"data_key": "data"},  # this goes to __call__
        #     "resource": "RES",
        #     ...: ...,
        # }
        # datum = {
        #     "datum_id": "b",
        #     "datum_kwargs": {"data_key": "omega"},
        #     "resource": "RES",
        #     ...: ...,
        # }

        # event = {...: ..., "data": {"detector_img": "a", "omega": "b"}}

        for data_key in self._datum_ids.keys():
            datum_id = f"{resource_uid}/{data_key}"
            self._datum_ids[data_key] = datum_id
            datum = {
                "resource": resource_uid,
                "datum_id": datum_id,
                "datum_kwargs": {"data_key": data_key},
            }
            asset_docs_cache.append(("datum", datum))
        return tuple(asset_docs_cache)

    def _extract_metadata(self, field="omega"):
        with h5py.File(self._master_file, "r") as hf:
            return hf.get(f"entry/sample/goniometer/{field}")[()]

    def unstage(self):
        ttime.sleep(1.0)
        super().unstage()

    def stage(self, *args, **kwargs):
        return super().stage(*args, **kwargs)

    def trigger(self, *args, **kwargs):
        status = super().trigger(*args, **kwargs)
        self.cam.special_trigger_button.set(1)
        return status

    def read(self, *args, streaming=False, **kwargs):
        """
        This is a test of using streaming read.
        Ideally, this should be handled by a new _stream_attrs property.
        For now, we just check for a streaming key in read and
        call super() if False, or read the one key we know we should read
        if True.

        Parameters
        ----------
        streaming : bool, optional
            whether to read streaming attrs or not
        """
        if streaming:
            key = self._image_name  # this comes from the SingleTrigger mixin
            read_dict = super().read()
            ret = OrderedDict({key: read_dict[key]})
            return ret
        ret = super().read(*args, **kwargs)
        return ret

    def describe(self, *args, streaming=False, **kwargs):
        """
        This is a test of using streaming read.
        Ideally, this should be handled by a new _stream_attrs property.
        For now, we just check for a streaming key in read and
        call super() if False, or read the one key we know we should read
        if True.

        Parameters
        ----------
        streaming : bool, optional
            whether to read streaming attrs or not
        """
        if streaming:
            key = self._image_name  # this comes from the SingleTrigger mixin
            read_dict = super().describe()
            ret = OrderedDict({key: read_dict[key]})
            return ret
        ret = super().describe(*args, **kwargs)
        return ret

    def super_unstage(self):
        super().unstage()


def set_eiger_defaults(eiger):
    """Choose which attributes to read per-step (read_attrs) or
    per-run (configuration attrs)."""

    eiger.read_attrs = [
        "file",
        # 'stats1', 'stats2', 'stats3', 'stats4', 'stats5',
    ]
    # for stats in [eiger.stats1, eiger.stats2, eiger.stats3,
    #               eiger.stats4, eiger.stats5]:
    #     stats.read_attrs = ['total']
    eiger.file.read_attrs = []
    eiger.cam.read_attrs = []
