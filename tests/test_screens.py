from bluesky import plans as bp
from cditools.screens import StandardProsilicaCam

def test_screenA1():
    screenA1 = StandardProsilicaCam() # prefix?
    yield from bp.count([screenA1], 2)

