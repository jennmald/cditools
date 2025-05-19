from datetime import datetime
from pathlib import PurePath
from typing import Any

from ophyd import EpicsSignalRO, ROIPlugin, Device, Component as Cpt, Signal, EigerDetector, StatusBase
from ophyd.areadetector.base import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.filestore_mixins import FileStoreBase, new_short_uid
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.plugins import StatsPlugin, ProcessPlugin, ROIPlugin


class EigerFileHandler(Device, FileStoreBase):
    """A device to handle the file writing for the Eiger detector.

    When the Eiger's FileWriter module and SaveFiles are enabled, the file writing is handled
    by the detector itself. In this case, we want to generate a resource document for the
    file path and file name pattern. Then, we want to generate a datum for each trigger that
    enables us to get the individual frames from the file.

    The alternative to this is to use the Stream interface and configure the area detector plugins
    to write to a file store.
    """
    sequence_id = ADComponent(EpicsSignalRO, 'SequenceId')
    file_path = ADComponent(EpicsSignalWithRBV, 'FilePath', string=True)
    file_write_name_pattern = ADComponent(EpicsSignalWithRBV, 'FWNamePattern',
                                          string=True)
    file_write_images_per_file = ADComponent(EpicsSignalWithRBV,
                                             'FWNImagesPerFile')
    current_run_start_uid = Cpt(Signal, value='', add_prefix=())

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        self.sequence_id_offset = 1
        super().__init__(*args, **kwargs)

        # NOTE: See `FileStoreBase._generate_resource` for the use of these.
        self._fn = None
        self.filestore_spec = "AD_EIGER2"

    def stage(self) -> list[object]:
        res_uid = new_short_uid()
        write_path = datetime.now().strftime(self.write_path_template)
        self.file_path.set(write_path).wait(1.0)
        # The name pattern must have `$id` in it.
        # `$id` is replaced by the current sequence id of the acquisition.
        # E.g. * <res_uid>_1_master.h5
        #      * <res_uid>_1_data_000001.h5
        #      * <res_uid>_1_data_000002.h5
        #      * ...
        self.file_write_name_pattern.set(f"{res_uid}_$id").wait(1.0)

        super().stage()

        # Set the filename for the resource document.
        file_prefix = PurePath(self.file_path.get()) / res_uid
        self._fn = file_prefix

        images_per_file = self.file_write_images_per_file.get()
        resource_kwargs = {'images_per_file' : images_per_file}

        self._generate_resource(resource_kwargs)

    def generate_datum(self, key: str, timestamp: float, datum_kwargs: dict[str, Any]) -> Any:
        # The detector keeps its own counter which is uses label HDF5
        # sub-files.  We access that counter via the sequence_id
        # signal and stash it in the datum.
        seq_id = self.sequence_id_offset + self.sequence_id.get()  # det writes to the NEXT one
        datum_kwargs.update({'seq_id': seq_id})
        # TODO: Is this needed?
        if self.frame_num is not None:
            datum_kwargs.update({'frame_num': self.frame_num})
        return super().generate_datum(key, timestamp, datum_kwargs)


class EigerBase(EigerDetector):
    """Base class for Eiger detectors that have the commonly used plugins."""
    file_handler = Cpt(EigerFileHandler, "", name="file_handler",
                       write_path_template="/nsls2/data/tst/legacy/mock-proposals/2025-2/pass-56789/assets/eiger/%Y/%m/%d",
                       root="/nsls2/data/tst/legacy/mock-proposals/2025-2/pass-56789/assets/eiger")
    stats1 = Cpt(StatsPlugin, "Stats1:")
    stats2 = Cpt(StatsPlugin, "Stats2:")
    stats3 = Cpt(StatsPlugin, "Stats3:")
    stats4 = Cpt(StatsPlugin, "Stats4:")
    stats5 = Cpt(StatsPlugin, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")
    

    def stage(self, *args: Any, **kwargs: dict[str, Any]) -> list[object]:
        staged_devices = super().stage(*args, **kwargs)
        self.cam.manual_trigger.set(1).wait(5.0)
        return staged_devices

    def unstage(self) -> None:
        self.cam.manual_trigger.set(0).wait(5.0)
        super().unstage()


class EigerSingleTrigger(SingleTrigger, EigerBase):
    """Eiger detector that uses the single trigger acquisition mode."""
    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.stage_sigs["cam.trigger_mode"] = 0
    
    def trigger(self, *args: Any, **kwargs: dict[str, Any]) -> StatusBase:
        status = super().trigger(*args, **kwargs)
        # If the manual trigger is enabled, we need to press the special trigger button
        # to actually trigger the detector.
        if self.cam.manual_trigger.get() == 1:
            self.cam.special_trigger_button.set(1).wait(5.0)
        return status
