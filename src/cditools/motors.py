from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from ophyd import Component as Cpt  # type: ignore[import-not-found]
from ophyd import (
    Device,
    EpicsMotor,
    PseudoPositioner,
    PseudoSingle,
    Signal,
)
from ophyd import DynamicDeviceComponent as DDC
from ophyd.pseudopos import (
    pseudo_position_argument,
    real_position_argument,
)
from scipy.interpolate import make_interp_spline


class EpicsMotorRO(EpicsMotor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def move(self, *args, **kwargs):  # noqa: ARG002
        msg = f"{self.name} is read-only and cannot be moved."
        raise PermissionError(msg)

    def stop(self, *args, **kwargs):  # noqa: ARG002
        msg = f"{self.name} is read-only and cannot be stopped manually."
        raise PermissionError(msg)

    def set(self, *args, **kwargs):  # noqa: ARG002
        msg = f"{self.name} is read-only and cannot be set."
        raise PermissionError(msg)

    def set_position(self, *args, **kwargs):  # noqa: ARG002
        msg = f"{self.name} is read-only and its position cannot be set."
        raise PermissionError(msg)

    def _readonly_put(self, *args, **kwargs):  # noqa: ARG002
        msg = f"{self.name} is read-only and cannot write PVs."
        raise PermissionError(msg)


class DM1(Device):
    """Ophyd Device for DM1"""

    slit = DDC(
        {
            "ib": (EpicsMotor, "Slt:WB1-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:WB1-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:WB1-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:WB1-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:WB1-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:WB1-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:WB1-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:WB1-Ax:VC}Mtr", {}),
        }
    )
    filt = Cpt(EpicsMotor, "Fltr:DM1-Ax:Y}Mtr")


class DMM(Device):
    h = Cpt(EpicsMotor, "Mono:DMM-Ax:TX}Mtr")
    v = Cpt(EpicsMotor, "Mono:DMM-Ax:TY}Mtr")
    bragg = Cpt(EpicsMotor, "Mono:DMM-Ax:Bragg}Mtr")
    mlm1 = DDC(
        {
            "r": (EpicsMotor, "Mono:DMM-Ax:Roll}Mtr", {}),
            "fr": (EpicsMotor, "Mono:DMM-Ax:FR}Mtr", {}),
        }
    )
    mgap = Cpt(EpicsMotor, "Mono:DMM-Ax:HG}Mtr")
    mlm2 = DDC(
        {
            "p": (EpicsMotor, "Mono:DMM-Ax:Pitch}Mtr", {}),
            "fp": (EpicsMotor, "Mono:DMM-Ax:FP}Mtr", {}),
        }
    )
    zoff = Cpt(EpicsMotor, "Mono:DMM-Ax:TZ}Mtr")


class DCMBase(Device):
    pitch = Cpt(EpicsMotor, "Mono:HDCM-Ax:Pitch}Mtr")
    fine: ClassVar[dict] = {
        "fpitch": Cpt(EpicsMotor, "Mono:HDCM-Ax:FP}Mtr"),
        "roll": Cpt(EpicsMotor, "Mono:HDCM-Ax:Roll}Mtr"),
    }
    h = Cpt(EpicsMotor, "Mono:HDCM-Ax:TX}Mtr")
    v = Cpt(EpicsMotor, "Mono:HDCM-Ax:TY}Mtr")


class Energy(PseudoPositioner):
    bragg = Cpt(EpicsMotor, "Mono:HDCM-Ax:Bragg}Mtr")
    cgap = Cpt(EpicsMotor, "Mono:HDCM-Ax:HG}Mtr")
    # Synthetic Axis
    energy = Cpt(PseudoSingle, egu="KeV")

    egu = Cpt(Signal, None, add_prefix=(), value="keV", kind="config")
    motor_egu = Cpt(Signal, None, add_prefix=(), value="eV", kind="config")

    ### not sure what PV should be used for the insertion device, example from SRX
    # u_gap = Cpt(InsertionDevice, "SR:C5-ID:G1{IVU21:1")
    _u_gap_offset = 0
    ### same for c2_x
    # c2_x = Cpt(EpicsMotor, "XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr", add_prefix=(), read_attrs=["user_readback"])
    # epics_d_spacing = EpicsSignal("XF:05IDA-CT{IOC:Status01}DCMDspacing.VAL")
    # epics_bragg_offset = EpicsSignal("XF:05IDA-CT{IOC:Status01}BraggOffset.VAL")

    # Energy "limits"
    _low = 5.0  # TODO: CHECK THIS VALUE, SRX uses 4.4
    _high = 15.0  # TODO: CHECK THIS VALUE, SRX uses 25

    # Set up constants
    Xoffset = 20.0  # mm
    d_111 = 3.1286911960950756
    ANG_OVER_KEV = 12.3984

    # Motor enable flags
    move_u_gap = Cpt(Signal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(Signal, None, add_prefix=(), value=True)
    harmonic = Cpt(Signal, None, add_prefix=(), value=0, kind="config")
    selected_harmonic = Cpt(Signal, None, add_prefix=(), value=0)

    # Experimental
    detune = Cpt(Signal, None, add_prefix=(), value=0)

    def __init__(
        self,
        *args,
        delta_bragg: int = 0,
        xoffset: int = 0,
        C2Xcal: int = 0,
        T2cal: int = 0,
        d_111: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._delta_bragg = delta_bragg
        self._xoffset = xoffset
        self._C2Xcal = C2Xcal
        self._T2cal = T2cal
        self._d_111 = d_111
        self.energy.readback.name = "energy"
        self.energy.setpoint.name = "energy_setpoint"
        calib_path = Path(__file__).parent
        # this is temporary, we need to figure out if there is a calib file for CDI
        calib_file = "../data/CDIUgapCalibration.txt"

        with Path.open(calib_path / calib_file) as f:
            next(f)
            uposlistIn = []
            elistIn = []
            for line in f:
                num = [float(x) for x in line.split()]
                # Check in case there is an extra line at the end of the calibration file
                if len(num) == 2:
                    uposlistIn.append(num[0])
                    elistIn.append(num[1])

        self.etoulookup = make_interp_spline(elistIn, uposlistIn)
        self.utoelookup = make_interp_spline(uposlistIn, elistIn)

        # can't do this until we know the insertion device
        # self.u_gap.gap.user_reacback.name = self.u_gap.name

    def energy_to_positions(
        self, target_energy: float, undulator_harmonic: int, u_detune: float
    ):
        """Compute undulator and mono positions given a target energy

        Parameters
        ----------
        target_energy : float
            Target energy in keV
        undulator_harmonic : int
            The harmonic in the undulator to use
        u_detune : float
            Amount to 'mistune' the undulator in keV

        Returns
        -------
        bragg : float
            The angle to set the monocromotor in radians
        gap : float
            The gap position in millimeters
        C2X : float
            The C2X position in millimeters
        ugap : float
            The undulator gap position in microns
        """

        # Calculate Bragg RBV
        bragg_RBV = (
            np.arcsin((self.ANG_OVER_KEV / target_energy) / (2 * self.d_111))
            - self._delta_bragg
        )
        bragg = bragg_RBV + self._delta_bragg
        T2 = self._xoffset + np.sin(bragg * np.pi / 180) / np.sin(
            2 * bragg * np.pi / 180
        )
        dT2 = T2 - self._T2cal
        C2X = self._C2Xcal - dT2

        # Calculate C2X
        gap = self._xoffset / 2 / np.cos(bragg)
        ugap = float(self.etoulookup((target_energy + u_detune) / undulator_harmonic))
        ugap *= 1000
        ugap = ugap + self._u_gap_offset

        return bragg, gap, C2X, ugap

    # def undulator_energy(self, harmonic: int = 3):
    # ugap = self.u_gap.gap.user_readback.get() / 1000

    # utoelookup = self.utoelookup
    # cannot do thids until we know ugap for insertion device
    # fundamental = float(utoelookup(ugap))
    # energy = fundamental * harmonic

    @pseudo_position_argument
    def forward(self, p_pos):
        energy = p_pos.energy  # energy assumed in keV
        bragg, gap = self.energy_to_positions(energy)
        harmonic = self.harmonic.get()
        if harmonic < 0 or ((harmonic % 2) == 0 and harmonic != 0):
            msg = f"The harmonic must be 0 or odd and positive, you set {harmonic}. Set `energy.harmonic` to a positive odd integer or 0."
            raise RuntimeError(msg)
        detune = self.detune.get()
        if energy <= self._low:
            msg = f"The energy you entered is too low ({energy} keV). "
            msg += f"Minimum energy = {self._low:.1f} keV"
            raise ValueError(msg)
        if energy > self._high:
            if (energy < self._low * 1000) or (energy > self._high * 1000):
                # Energy is invalid
                msg = f"The requested photon energy is invalid ({energy} keV). "
                msg += f"Values must be in the range of {self._low:.1f} - {self._high:.1f} keV"
                raise ValueError(msg)
            # Energy is in eV
            energy = energy / 1000.0

        if harmonic < 3:
            harmonic = 3
            # Choose the right harmonic
            braggcal, c2xcal, ugapcal = self.energy_to_positions(
                energy, harmonic, detune
            )
            # Try higher harmonics until the required gap is too small
            while True:
                braggcal, c2xcal, ugapcal = self.energy_to_positions(
                    energy, harmonic + 2, detune
                )
                if ugapcal < self.u_gap.low_limit:
                    break
                harmonic += 2

        self.selected_harmonic.put(harmonic)

        # Compute where we would move everything to in a perfect world
        bragg, c2_x, u_gap = self.energy_to_positions(energy, harmonic, detune)

        # Sometimes move the crystal gap
        if not self.move_c2_x.get():
            c2_x = self.c2_x.position

        # Sometimes move the undulator
        if not self.move_u_gap.get():
            u_gap = self.u_gap.position

        return self.RealPosition(bragg=np.rad2deg(bragg), c2_x=c2_x, cgap=u_gap)

    @real_position_argument
    def inverse(self, r_pos):
        bragg = np.deg2rad(r_pos.bragg)
        e = self.ANG_OVER_KEV / (2 * self.d_111 * np.sin(bragg))
        return self.PseudoPosition(energy=float(e))

    @pseudo_position_argument
    def set(self, position):
        return super().set([float(_) for _ in position])

    def sync_with_epics(self):
        self.epics_d_spacing.put(self._d_111)
        self.epics_bragg_offset.put(self._delta_bragg)

    def retune_undulator(self):
        self.detune.put(0.0)
        self.move(self.engergy.get()[0])

    # def mono_peakup(element, acquisition_time=1.0, peakup=True):
    #     """
    #         First draft of the mono peakup scan
    #         Need more info about the axis to be scanned, the move ID, and which detector will be used for feedback.
    #     Args:
    #         element (string): element name
    #         acquisition_time (float, optional): _description_. Defaults to 1.0.
    #         peakup (bool, optional): _description_. Defaults to True.
    #     """
    #     getemissionE(element)
    #     energy_x = getbindingsE(element)

    #     yield from mov(energy, energy_x)
    #     setroi(1, element)
    #     if peakup:
    #         yield from bps.sleep(5)
    #         yield from peakup()
    #     yield from xanes_plan(
    #         erange=[energy_x - 100, energy_x + 50],
    #         estep=[1.0],
    #         samplename=f"{element}Foil",
    #         filename=f"{element}Foilstd",
    #         acqtime=acquisition_time,
    #         shutter=True,
    #     )


class VPM(Device):
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:VPM-Ax:Y}Mtr", {}),
        }
    )

    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:VPM-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:VPM-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:VPM-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:VPM-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:VPM-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:VPM-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:VPM-Ax:YDO}Mtr", {}),
            "p": (EpicsMotor, "Mir:VPM-Ax:Pitch}Mtr", {}),
            "r": (EpicsMotor, "Mir:VPM-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:VPM-Ax:TY}Mtr", {}),
            "x": (EpicsMotorRO, "Mir:VPM-Ax:TX}Mtr", {}),
            "yaw": (EpicsMotorRO, "Mir:VPM-Ax:Yaw}Mtr", {}),
            "us_lt": (EpicsMotorRO, "Mir:VPM-Ax:XU}Mtr", {}),
            "ds_lt": (EpicsMotorRO, "Mir:VPM-Ax:XD}Mtr", {}),
            "us_b": (EpicsMotorRO, "Mir:VPM-Ax:UB}Mtr", {}),
            "ds_b": (EpicsMotorRO, "Mir:VPM-Ax:DB}Mtr", {}),
            "bend": (EpicsMotorRO, "Mir:VPM-Ax:Bnd}Mtr", {}),
            "bend_off": (EpicsMotorRO, "Mir:VPM-Ax:BndOff}Mtr", {}),
        }
    )


class HPM(Device):
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:HPM-Ax:Y}Mtr", {}),
        }
    )

    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:HPM-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:HPM-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:HPM-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:HPM-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:HPM-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:HPM-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:HPM-Ax:YDO}Mtr", {}),
            "p": (EpicsMotor, "Mir:HPM-Ax:Pitch}Mtr", {}),
            "r": (EpicsMotor, "Mir:HPM-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:HPM-Ax:TY}Mtr", {}),
            "bend": (EpicsMotor, "Mir:HPM-Ax:Bnd}Mtr", {}),
            "bend_off": (EpicsMotor, "Mir:HPM-Ax:BndOff}Mtr", {}),
            "us_x": (EpicsMotor, "Mir:HPM-Ax:XU}Mtr", {}),
            "ds_x": (EpicsMotor, "Mir:HPM-Ax:XD}Mtr", {}),
            "x": (EpicsMotor, "Mir:HPM-Ax:TX}Mtr", {}),
            "yaw": (EpicsMotorRO, "Mir:HPM-Ax:Yaw}Mtr", {}),
            "us_lt": (EpicsMotorRO, "Mir:HPM-Ax:XU}Mtr", {}),
            "ds_lt": (EpicsMotorRO, "Mir:HPM-Ax:XD}Mtr", {}),
            "us_b": (EpicsMotorRO, "Mir:HPM-Ax:UB}Mtr", {}),
            "ds_b": (EpicsMotorRO, "Mir:HPM-Ax:DB}Mtr", {}),
        }
    )


class DM2(Device):
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:DM2-Ax:Y}Mtr", {}),
        }
    )
    foil = Cpt(EpicsMotor, "IM:DM2-Ax:Y}Mtr")


class DM3(Device):
    slit = DDC(
        {
            "ib": (EpicsMotor, "Slt:DM3-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:DM3-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:DM3-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:DM3-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:DM3-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:DM3-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:DM3-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:DM3-Ax:VC}Mtr", {}),
        }
    )
    bpm = DDC(
        {
            "x": (EpicsMotor, "BPM:DM3-Ax:TX}Mtr", {}),
            "y": (EpicsMotor, "BPM:DM3-Ax:TY}Mtr", {}),
            "foil": (EpicsMotor, "BPM:DM3-Ax:Foil}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:DM3-Ax:FS}Mtr", {}),
        }
    )


class VKB(Device):
    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:KBv-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:KBv-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:KBv-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:KBv-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:KBv-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:KBv-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:KBv-Ax:YDO}Mtr", {}),
            "yaw": (EpicsMotor, "Mir:KBv-Ax:Yaw}Mtr", {}),
            "r": (EpicsMotor, "Mir:KBv-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:KBv-Ax:TY}Mtr", {}),
            "x": (EpicsMotor, "Mir:KBv-Ax:TX}Mtr", {}),
            "z": (EpicsMotor, "Mir:KBv-Ax:TZ}Mtr", {}),
            "p": (EpicsMotor, "Mir:KBv-Ax:Pitch}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "Mir:KBv-Ax:FS}Mtr", {}),
        }
    )


class HKB(Device):
    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:KBh-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:KBh-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:KBh-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:KBh-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:KBh-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:KBh-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:KBh-Ax:YDO}Mtr", {}),
            "yaw": (EpicsMotor, "Mir:KBh-Ax:Yaw}Mtr", {}),
            "r": (EpicsMotor, "Mir:KBh-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:KBh-Ax:TY}Mtr", {}),
            "x": (EpicsMotor, "Mir:KBh-Ax:TX}Mtr", {}),
            "z": (EpicsMotor, "Mir:KBh-Ax:TZ}Mtr", {}),
            "p": (EpicsMotor, "Mir:KBh-Ax:Pitch}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "Mir:KBh-Ax:FS}Mtr", {}),
        }
    )


class KB(Device):
    vkb = Cpt(VKB, "")
    hkb = Cpt(HKB, "")
    win = DDC(
        {
            "x": (EpicsMotor, "Wnd:Exit-Ax:TX}Mtr", {}),
            "y": (EpicsMotor, "Wnd:Exit-Ax:TY}Mtr", {}),
        }
    )


class DM4(Device):
    bpm = DDC(
        {
            "x": (EpicsMotor, "BPM:DM4-Ax:TX}Mtr", {}),
            "y": (EpicsMotor, "BPM:DM4-Ax:TY}Mtr", {}),
        }
    )


class SAM(Device):
    c_sm = DDC(
        {
            "lrx": (EpicsMotor, "Gon:1-Ax:Rx1}Mtr", {}),
            "lrz": (EpicsMotor, "Gon:1-Ax:Rz1}Mtr", {}),
        }
    )
    c_lg = DDC(
        {
            "lrx": (EpicsMotor, "Gon:1-Ax:Rx2}Mtr", {}),
            "lrz": (EpicsMotor, "Gon:1-Ax:Rz2}Mtr", {}),
        }
    )
    ly = Cpt(EpicsMotor, "Gon:1-Ax:Y}Mtr")
    ry = Cpt(EpicsMotor, "Gon:1-Ax:Ry}Mtr")
    t_sm = DDC(
        {
            "lx": (EpicsMotor, "Gon:1-Ax:X1}Mtr", {}),
            "lz": (EpicsMotor, "Gon:1-Ax:Z1}Mtr", {}),
        }
    )
    t_lg = DDC(
        {
            "lx": (EpicsMotor, "Gon:1-Ax:X2}Mtr", {}),
            "lz": (EpicsMotor, "Gon:1-Ax:Z2}Mtr", {}),
        }
    )
    lfx = Cpt(EpicsMotor, "Gon:1-Ax:XP}Mtr")
    lfy = Cpt(EpicsMotor, "Gon:1-Ax:YP}Mtr")
    lfz = Cpt(EpicsMotor, "Gon:1-Ax:ZP}Mtr")


class GON(Device):
    sam = Cpt(SAM, "")
    align = DDC(
        {
            "rx": (EpicsMotor, "Gon:1-Ax:Rx3}Mtr", {}),
            "rz": (EpicsMotor, "Gon:1-Ax:Rz3}Mtr", {}),
            "x": (EpicsMotor, "Gon:1-Ax:X3}Mtr", {}),
            "y": (EpicsMotor, "Gon:1-Ax:Y3}Mtr", {}),
            "z": (EpicsMotor, "Gon:1-Ax:Z3}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "Gon:1-Ax:Visual}Mtr", {}),
        }
    )


class BCU(Device):
    slit_us = DDC(
        {
            "ib": (EpicsMotor, "Slt:BCUU-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:BCUU-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:BCUU-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:BCUU-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:BCUU-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:BCUU-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:BCUU-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:BCUU-Ax:VC}Mtr", {}),
        }
    )
    slit_ds = DDC(
        {
            "ib": (EpicsMotor, "Slt:BCUD-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:BCUD-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:BCUD-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:BCUD-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:BCUD-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:BCUD-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:BCUD-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:BCUD-Ax:VC}Mtr", {}),
        }
    )
    ilcam = DDC(
        {
            "x": (EpicsMotor, "Qstar:1-Ax:1}Mtr", {}),
            "y": (EpicsMotor, "Qstar:1-Ax:2}Mtr", {}),
            "z": (EpicsMotor, "Qstar:1-Ax:3}Mtr", {}),
        }
    )
