
import os
import sys
from caproto.server import run
from cditools.simulated.black_hole import CDIBlackHoleIOC

# Set environment variables in the subprocess
os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

# Run the IOC
run(CDIBlackHoleIOC().pvdb, interfaces=["127.0.0.1"])
