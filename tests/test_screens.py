from __future__ import annotations

from bluesky import plans as bp

from cditools.screens import StandardProsilicaCam


def test_cam_A1():
    cam_A1 = StandardProsilicaCam("XF:09IDA-BI{DM:1-Cam:1}", name="cam_A1")
    cam_A1.wait_for_connection(timeout=60.0)
    yield from bp.count([cam_A1], 1)


def test_cam_A2():
    cam_A2 = StandardProsilicaCam("XF:09IDA-BI:1{WBStop-Cam:2}", name="cam_A2")
    cam_A2.wait_for_connection(timeout=60.0)
    yield from bp.count([cam_A2], 1)


def test_VPM_screen():
    cam_VPM = StandardProsilicaCam("XF:09IDB-BI:1{VPM-Cam:3}", name="cam_A3")
    cam_VPM.wait_for_connection(timeout=60.0)
    yield from bp.count([cam_VPM], 1)


def test_HPM_screen():
    cam_HPM = StandardProsilicaCam("XF:09IDB-BI:1{HPM-Cam:4}", name="cam_A4")
    cam_HPM.wait_for_connection(timeout=60.0)
    yield from bp.count([cam_HPM], 1)
