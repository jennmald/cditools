#!/usr/bin/env python
from ophyd import Device, Component as Cpt, EpicsMotor

# Auto-generated Ophyd device classes

class SltWB1(Device):
    """Ophyd Device for Slt WB1"""
    i = Cpt(EpicsMotor, '-Ax:I}Mtr')
    o = Cpt(EpicsMotor, '-Ax:O}Mtr')
    b = Cpt(EpicsMotor, '-Ax:B}Mtr')
    t = Cpt(EpicsMotor, '-Ax:T}Mtr')
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    _default_read_attrs = ['i', 'o', 'b', 't', 'hg', 'hc', 'vg', 'vc']

class FltrDM1(Device):
    """Ophyd Device for Fltr DM1"""
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    _default_read_attrs = ['y']


class FSVPM(Device):
    """Ophyd Device for FS VPM"""
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    _default_read_attrs = ['y']


class FSHPM(Device):
    """Ophyd Device for FS HPM"""
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    _default_read_attrs = ['y']


class FSDM2(Device):
    """Ophyd Device for FS DM2"""
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    _default_read_attrs = ['y']


class MonoDMM(Device):
    """Ophyd Device for Mono DMM"""
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    bragg = Cpt(EpicsMotor, '-Ax:Bragg}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:Roll}Mtr')
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:Pitch}Mtr')
    tz = Cpt(EpicsMotor, '-Ax:TZ}Mtr')
    fp = Cpt(EpicsMotor, '-Ax:FP}Mtr')
    fr = Cpt(EpicsMotor, '-Ax:FR}Mtr')
    _default_read_attrs = ['tx', 'ty', 'bragg', 'roll', 'hg', 'pitch', 'tz', 'fp', 'fr']


class MonoHDCM(Device):
    """Ophyd Device for Mono HDCM"""
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    bragg = Cpt(EpicsMotor, '-Ax:Bragg}Mtr')
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:Pitch}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:Roll}Mtr')
    fp = Cpt(EpicsMotor, '-Ax:FP}Mtr')
    _default_read_attrs = ['tx', 'ty', 'bragg', 'hg', 'pitch', 'roll', 'fp']


class SltVPM(Device):
    """Ophyd Device for Slt VPM"""
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    _default_read_attrs = ['hg', 'hc', 'vg', 'vc']


class SltHPM(Device):
    """Ophyd Device for Slt HPM"""
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    _default_read_attrs = ['hg', 'hc', 'vg', 'vc']


class MirVPM(Device):
    """Ophyd Device for Mir VPM"""
    yuc = Cpt(EpicsMotor, '-Ax:YUC}Mtr')
    ydi = Cpt(EpicsMotor, '-Ax:YDI}Mtr')
    ydo = Cpt(EpicsMotor, '-Ax:YDO}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:Pitch}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:Roll}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    ub = Cpt(EpicsMotor, '-Ax:UB}Mtr')
    db = Cpt(EpicsMotor, '-Ax:DB}Mtr')
    bnd = Cpt(EpicsMotor, '-Ax:Bnd}Mtr')
    bndoff = Cpt(EpicsMotor, '-Ax:BndOff}Mtr')
    _default_read_attrs = ['yuc', 'ydi', 'ydo', 'pitch', 'roll', 'ty', 'tx', 'ub', 'db', 'bnd', 'bndoff']


class IMDM2(Device):
    """Ophyd Device for IM DM2"""
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    _default_read_attrs = ['y']


class MirHPM(Device):
    """Ophyd Device for Mir HPM"""
    yuc = Cpt(EpicsMotor, '-Ax:YUC}Mtr')
    ydi = Cpt(EpicsMotor, '-Ax:YDI}Mtr')
    ydo = Cpt(EpicsMotor, '-Ax:YDO}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:Pitch}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:Roll}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    ub = Cpt(EpicsMotor, '-Ax:UB}Mtr')
    db = Cpt(EpicsMotor, '-Ax:DB}Mtr')
    bnd = Cpt(EpicsMotor, '-Ax:Bnd}Mtr')
    bndoff = Cpt(EpicsMotor, '-Ax:BndOff}Mtr')
    xu = Cpt(EpicsMotor, '-Ax:XU}Mtr')
    xd = Cpt(EpicsMotor, '-Ax:XD}Mtr')
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    yaw = Cpt(EpicsMotor, '-Ax:Yaw}Mtr')
    _default_read_attrs = ['yuc', 'ydi', 'ydo', 'pitch', 'roll', 'ty', 'ub', 'db', 'bnd', 'bndoff', 'xu', 'xd', 'tx', 'yaw']


class SltDM3(Device):
    """Ophyd Device for Slt DM3"""
    i = Cpt(EpicsMotor, '-Ax:I}Mtr')
    o = Cpt(EpicsMotor, '-Ax:O}Mtr')
    b = Cpt(EpicsMotor, '-Ax:B}Mtr')
    t = Cpt(EpicsMotor, '-Ax:T}Mtr')
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    _default_read_attrs = ['i', 'o', 'b', 't', 'hg', 'hc', 'vg', 'vc']


class BPMDM3(Device):
    """Ophyd Device for BPM DM3"""
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    foil = Cpt(EpicsMotor, '-Ax:Foil}Mtr')
    _default_read_attrs = ['tx', 'ty', 'foil']


class FSDM3(Device):
    """Ophyd Device for FS DM3"""
    fs = Cpt(EpicsMotor, '-Ax:FS}Mtr')
    _default_read_attrs = ['fs']


class MirKBv(Device):
    """Ophyd Device for Mir KBv"""
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    yuc = Cpt(EpicsMotor, '-Ax:YUC}Mtr')
    ydi = Cpt(EpicsMotor, '-Ax:YDI}Mtr')
    ydo = Cpt(EpicsMotor, '-Ax:YDO}Mtr')
    yaw = Cpt(EpicsMotor, '-Ax:Yaw}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:Roll}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    tz = Cpt(EpicsMotor, '-Ax:TZ}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:Pitch}Mtr')
    fs = Cpt(EpicsMotor, '-Ax:FS}Mtr')
    _default_read_attrs = ['hg', 'hc', 'vg', 'vc', 'yuc', 'ydi', 'ydo', 'yaw', 'roll', 'ty', 'tx', 'tz', 'pitch', 'fs']


class MirKBh(Device):
    """Ophyd Device for Mir KBh"""
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    yuc = Cpt(EpicsMotor, '-Ax:YUC}Mtr')
    ydi = Cpt(EpicsMotor, '-Ax:YDI}Mtr')
    ydo = Cpt(EpicsMotor, '-Ax:YDO}Mtr')
    yaw = Cpt(EpicsMotor, '-Ax:Yaw}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:Roll}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    tz = Cpt(EpicsMotor, '-Ax:TZ}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:Pitch}Mtr')
    fs = Cpt(EpicsMotor, '-Ax:FS}Mtr')
    _default_read_attrs = ['hg', 'hc', 'vg', 'vc', 'yuc', 'ydi', 'ydo', 'yaw', 'roll', 'ty', 'tx', 'tz', 'pitch', 'fs']


class BPMDM4(Device):
    """Ophyd Device for BPM DM4"""
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    _default_read_attrs = ['tx', 'ty']


class WndExit(Device):
    """Ophyd Device for Wnd Exit"""
    tx = Cpt(EpicsMotor, '-Ax:TX}Mtr')
    ty = Cpt(EpicsMotor, '-Ax:TY}Mtr')
    _default_read_attrs = ['tx', 'ty']


class Gon1(Device):
    """Ophyd Device for Gon 1"""
    rx1 = Cpt(EpicsMotor, '-Ax:Rx1}Mtr')
    rz1 = Cpt(EpicsMotor, '-Ax:Rz1}Mtr')
    rx2 = Cpt(EpicsMotor, '-Ax:Rx2}Mtr')
    rz2 = Cpt(EpicsMotor, '-Ax:Rz2}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    ry = Cpt(EpicsMotor, '-Ax:Ry}Mtr')
    x1 = Cpt(EpicsMotor, '-Ax:X1}Mtr')
    z1 = Cpt(EpicsMotor, '-Ax:Z1}Mtr')
    x2 = Cpt(EpicsMotor, '-Ax:X2}Mtr')
    z2 = Cpt(EpicsMotor, '-Ax:Z2}Mtr')
    rx3 = Cpt(EpicsMotor, '-Ax:Rx3}Mtr')
    rz3 = Cpt(EpicsMotor, '-Ax:Rz3}Mtr')
    x3 = Cpt(EpicsMotor, '-Ax:X3}Mtr')
    y3 = Cpt(EpicsMotor, '-Ax:Y3}Mtr')
    z3 = Cpt(EpicsMotor, '-Ax:Z3}Mtr')
    visual = Cpt(EpicsMotor, '-Ax:Visual}Mtr')
    xp = Cpt(EpicsMotor, '-Ax:XP}Mtr')
    yp = Cpt(EpicsMotor, '-Ax:YP}Mtr')
    zp = Cpt(EpicsMotor, '-Ax:ZP}Mtr')
    _default_read_attrs = ['rx1', 'rz1', 'rx2', 'rz2', 'y', 'ry', 'x1', 'z1', 'x2', 'z2', 'rx3', 'rz3', 'x3', 'y3', 'z3', 'visual', 'xp', 'yp', 'zp']


class SltBCUU(Device):
    """Ophyd Device for Slt BCUU"""
    i = Cpt(EpicsMotor, '-Ax:I}Mtr')
    o = Cpt(EpicsMotor, '-Ax:O}Mtr')
    b = Cpt(EpicsMotor, '-Ax:B}Mtr')
    t = Cpt(EpicsMotor, '-Ax:T}Mtr')
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    _default_read_attrs = ['i', 'o', 'b', 't', 'hg', 'hc', 'vg', 'vc']


class SltBCUD(Device):
    """Ophyd Device for Slt BCUD"""
    i = Cpt(EpicsMotor, '-Ax:I}Mtr')
    o = Cpt(EpicsMotor, '-Ax:O}Mtr')
    b = Cpt(EpicsMotor, '-Ax:B}Mtr')
    t = Cpt(EpicsMotor, '-Ax:T}Mtr')
    hg = Cpt(EpicsMotor, '-Ax:HG}Mtr')
    hc = Cpt(EpicsMotor, '-Ax:HC}Mtr')
    vg = Cpt(EpicsMotor, '-Ax:VG}Mtr')
    vc = Cpt(EpicsMotor, '-Ax:VC}Mtr')
    _default_read_attrs = ['i', 'o', 'b', 't', 'hg', 'hc', 'vg', 'vc']


class Qstar1(Device):
    """Ophyd Device for Qstar 1"""
    ax1 = Cpt(EpicsMotor, '-Ax:1}Mtr')
    ax2 = Cpt(EpicsMotor, '-Ax:2}Mtr')
    ax3 = Cpt(EpicsMotor, '-Ax:3}Mtr')
    _default_read_attrs = ['ax1', 'ax2', 'ax3']
