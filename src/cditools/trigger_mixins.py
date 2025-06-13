class CDIModalTrigger(HxnModalBase, TriggerBase):
    def __init__(self, *args, image_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        if image_name is None:
            image_name = '_'.join([self.name, 'image'])
        self._image_name = image_name
        self._external_acquire_at_stage = True

    def stop(self, success=False):
        ret = super().stop(success=success)
        self._acquisition_signal.put(0, wait=True)
        return ret

    def mode_internal(self):
        super().mode_internal()

        cam = self.cam
        cam.stage_sigs[cam.acquire] = 0
        ordered_dict_move_to_beginning(cam.stage_sigs, cam.acquire)

        cam.stage_sigs[cam.num_images] = 1
        cam.stage_sigs[cam.image_mode] = 'Single'
        cam.stage_sigs[cam.trigger_mode] = 'Internal'

    def mode_external(self):
        super().mode_external()
        total_points = self.mode_settings.total_points.get()

        cam = self.cam
        cam.stage_sigs[cam.num_images] = total_points
        cam.stage_sigs[cam.image_mode] = 'Multiple'
        cam.stage_sigs[cam.trigger_mode] = 'External'

    def stage(self):
        self._acquisition_signal.subscribe(self._acquire_changed)
        staged = super().stage()

        # In external triggering mode, the devices is only triggered once at
        # stage
        if self.mode == 'external' and self._external_acquire_at_stage:
            self._acquisition_signal.put(1, wait=False)
        return staged

    def unstage(self):
        try:
            return super().unstage()
        finally:
            self._acquisition_signal.clear_sub(self._acquire_changed)

    def trigger_internal(self):
        if self._staged != Staged.yes:
            raise RuntimeError("This detector is not ready to trigger."
                               "Call the stage() method before triggering.")

        self._status = DeviceStatus(self)
        self._acquisition_signal.put(1, wait=False)
        self.dispatch(self._image_name, ttime.time())
        return self._status

    def trigger_external(self):
        if self._staged != Staged.yes:
            raise RuntimeError("This detector is not ready to trigger."
                               "Call the stage() method before triggering.")

        self._status = DeviceStatus(self)
        self._status._finished()
        # TODO this timestamp is inaccurate!
        if self.mode_settings.scan_type.get() != 'fly':
            # Don't dispatch images for fly-scans - they are bulk read at the end
            self.dispatch(self._image_name, ttime.time())
        return self._status

    def trigger(self):
        mode_trigger = getattr(self, f'trigger_{self.mode}')
        return mode_trigger()

    def _acquire_changed(self, value=None, old_value=None, **kwargs):
        '''This is called when the 'acquire' signal changes.'''
        if self._status is None:
            return
        if (old_value == 1) and (value == 0):
            # Negative-going edge means an acquisition just finished.
            self._status._finished()


class FileStoreBulkReadable(FileStoreIterativeWrite):

    def _reset_data(self):
        self._datum_uids.clear()
        self._point_counter = itertools.count()

    def bulk_read(self, timestamps):
        image_name = self.image_name

        uids = [self.generate_datum(self.image_name, ts, {}) for ts in timestamps]

        # clear so unstage will not save the images twice:
        self._reset_data()
        return {image_name: uids}

    @property
    def image_name(self):
        return self.parent._image_name