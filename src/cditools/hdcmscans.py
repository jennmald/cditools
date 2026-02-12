import numpy as np
import matplotlib.pyplot as plt
import bluesky.plan_stubs as bps
import skbeam.core.constants.xrf as xrfC

interestinglist = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                   'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                   'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                   'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                   'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                   'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                   'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                   'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

elements = dict()
element_edges = ['ka1', 'ka2', 'kb1', 'la1', 'la2', 'lb1', 'lb2', 'lg1', 'ma1']
element_transitions = ['k', 'l1', 'l2', 'l3', 'm1', 'm2', 'm3', 'm4', 'm5']
for i in interestinglist:
    elements[i] = xrfC.XrfElement(i)

def getemissionE(element: str, edge: str | None = None) -> float | None: 
    cur_element = xrfC.XrfElement(element)
    if edge is None:
        print("Edge\tEnergy [keV]")
        for e in element_edges:
            if cur_element.emission_line[e] < 25. and \
               cur_element.emission_line[e] > 1.:
                # print("{0:s}\t{1:8.2f}".format(e, cur_element.emission_line[e]))
                print(f"{e}\t{cur_element.emission_line[e]:8.2f}")
    else:
        return np.round(cur_element.emission_line[edge], 3)


def getbindingsE(element, edge=None):
    if edge is None:
        y = [0., 'k']
        print("Edge\tEnergy [eV]\tYield")
        for i in ['k', 'l1', 'l2', 'l3']:
            print(f"{i}\t"
                  f"{xrfC.XrayLibWrap(elements[element].Z,'binding_e')[i]*1000.:8.2f}\t"
                  f"{xrfC.XrayLibWrap(elements[element].Z,'yield')[i]:5.3f}")
            if (y[0] < xrfC.XrayLibWrap(elements[element].Z, 'yield')[i] and
               xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[i] < 25.):
                y[0] = xrfC.XrayLibWrap(elements[element].Z, 'yield')[i]
                y[1] = i
        return np.round(xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[y[1]] * 1000., 3)
    else:
        return np.round(xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[edge] * 1000., 3)


def setroi(roinum: int, element: str, edge: str | None = None, det: object | None = None):
    '''
    Set energy ROIs for Vortex SDD.
    Selects elemental edge given current energy if not provided.
    element     <symbol>    element symbol for target energy
    edge                    optional:  ['ka1', 'ka2', 'kb1', 'la1', 'la2',
                                        'lb1', 'lb2', 'lg1', 'ma1']
    det                     optional: detector object
    '''
    cur_element = xrfC.XrfElement(element)
    if edge is None:
        for e in ['ka1', 'ka2', 'kb1', 'la1', 'la2',
                  'lb1', 'lb2', 'lg1', 'ma1']:
            if cur_element.emission_line[e] < energy.energy.get()[1]:
                edge = 'e'
                break
    else:
        e = edge

    e_ch = int(cur_element.emission_line[e] * 1000)
    if det is not None:
        channels = [det.channels.channel01, ]
    else:
        channels = list(xs.iterate_channels())
    

def mono_peakup(element, acquisition_time=1.0, peakup=True):
    """ 
        First draft of the mono peakup scan
        Need more info about the axis to be scanned, the move ID, and which detector will be used for feedback.
    Args:
        element (string): element name 
        acquisition_time (float, optional): _description_. Defaults to 1.0.
        peakup (bool, optional): _description_. Defaults to True.
    """
    getemissionE(element)
    energy_x = getbindingsE(element)

    yield from mov(energy, energy_x)
    setroi(1, element)
    if peakup:
        yield from bps.sleep(5)
        yield from peakup()
    yield from xanes_plan(erange=[energy_x-100,energy_x+50],
                          estep=[1.0],
                          samplename=f'{element}Foil',
                          filename=f'{element}Foilstd',
                          acqtime=acquisition_time,
                          shutter=True)