from __future__ import annotations

import os
import time
from collections.abc import Generator
from subprocess import PIPE, Popen

import pytest
from bluesky.plans import scan
from ophyd import Device, EpicsSignal

from cditools.motors import (
    BCU,
    DCM,
    DM1,
    DM2,
    DM3,
    DM4,
    DMM,
    GON,
    HPM,
    KB,
    VPM,
)

EpicsSignal.set_defaults(timeout=20.0, connection_timeout=20.0, write_timeout=20.0)
Device.set_defaults(connection_timeout=20.0)


@pytest.fixture(scope="session")
def black_hole_ioc() -> Generator[None, None, None]:
    os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    p = Popen(["black-hole-ioc", "--interfaces", "127.0.0.1"], stdout=PIPE)
    if not p.stdout:
        msg = "Failed to start black-hole-ioc"
        raise RuntimeError(msg)
    start_time = time.time()
    timeout = 30  # seconds
    while True:
        line = p.stdout.readline().decode("utf-8")
        if line.strip().endswith("Server startup complete."):
            break
        if time.time() - start_time > timeout:
            p.terminate()
            msg = "Timeout waiting for black-hole-ioc to start"
            raise RuntimeError(msg)
        if not line:  # Process ended without expected output
            msg = "black-hole-ioc process ended unexpectedly"
            raise RuntimeError(msg)
        time.sleep(0.1)  # Small delay to prevent CPU spinning
    yield
    p.terminate()
    p.wait()


def test_motors_can_connect(black_hole_ioc: None) -> None:
    dm1 = DM1(prefix="XF:09IDA-OP:1{", name="dm1")
    dm1.wait_for_connection(timeout=60.0)

    vpm = VPM(prefix="XF:09IDA-OP:1{", name="vpm")
    vpm.wait_for_connection(timeout=60.0)

    hpm = HPM(prefix="XF:09IDA-OP:1{", name="hpm")
    hpm.wait_for_connection(timeout=60.0)

    dm2 = DM2(prefix="XF:09IDA-OP:1{", name="dm2")
    dm2.wait_for_connection(timeout=60.0)

    dmm = DMM(prefix="XF:09IDA-OP:1{", name="dmm")
    dmm.wait_for_connection(timeout=60.0)

    dcm = DCM(prefix="XF:09IDA-OP:1{", name="dcm")
    dcm.wait_for_connection(timeout=60.0)

    dm3 = DM3(prefix="XF:09IDB-OP:1{", name="dm3")
    dm3.wait_for_connection(timeout=60.0)

    kb = KB(prefix="XF:09IDC-OP:1{", name="kb")
    kb.wait_for_connection(timeout=60.0)

    dm4 = DM4(prefix="XF:09IDC-OP:1{", name="dm4")
    dm4.wait_for_connection(timeout=60.0)

    gon = GON(prefix="XF:09IDC-OP:1{", name="gon")
    gon.wait_for_connection(timeout=60.0)

    bcu = BCU(prefix="XF:09IDC-OP:1{", name="bcu")
    bcu.wait_for_connection(timeout=60.0)


def test_DCM_scan():
    dcm = DCM(prefix="XF:09IDA-OP:1{", name="dcm", labels=["motors"])
    dcm.wait_for_connection(timeout=60.0)
    RE(scan([], dcm.h, -3, 3, 2))

def test_HPM_slits():
    hpm = HPM(prefix="XF:09IDA-OP:1{", name="hpm", labels=["motors"])
    hpm.wait_for_connection(timeout=60.0)
    # horizontal gap
    RE(scan([], hpm.slits.hg, 1.2, 1.4, 2))
    # vertical gap
    RE(scan([], hpm.slits.vg, 1.2, 1.4, 2))
    # horizontal center
    RE(scan([], hpm.slits.hc, -0.1, 0.1, 2))
    # vertical center
    RE(scan([], hpm.slits.vc, -0.1, 0.1, 2))

def test_VPM_slits():
    vpm = VPM(prefix="XF:09IDA-OP:1{", name="vpm", labels=["motors"])
    vpm.wait_for_connection(timeout=60.0)
    # horizontal gap
    RE(scan([], vpm.slits.hg, 1.2, 1.4, 2))
    # vertical gap
    RE(scan([], vpm.slits.vg, 1.2, 1.4, 2))
    # horizontal center
    RE(scan([], vpm.slits.hc, -0.1, 0.1, 2))
    # vertical center
    RE(scan([], vpm.slits.vc, -0.1, 0.1, 2))