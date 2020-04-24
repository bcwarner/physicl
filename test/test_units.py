import pytest

import sys
import os

import phys
import phys.newton
import phys.light
import numpy as np
import numpy.linalg as lin
import time

# Tests whether the non-zero entries in a dictionary are equal
# TODO: Maybe move this into the ufunc handler.
# Note: For tests, make sure to put in manual calculations for all tests.
def dict_equiv(a, b):
	for k, v in a.items():
		if k in b and b[k] != 0 and v != b[k]:
			return False
	for k, v in b.items():
		if k in a and a[k] != 0 and v != a[k]:
			return False
	return True

def test_units_1():
	x = phys.Measurement(5, "kg**1 m**1 s**-2")
	y = phys.Measurement(5, "N**1")
	assert x == y
	assert x.scale == x.scale
	assert x.units == x.units

# Tests whether scaling is correctly handled.
def test_units_2():
	x = phys.Measurement(1, "au**1")
	y = phys.Measurement(149597870700 * 1, "m**1")
	assert x + y == phys.Measurement(2, "au**1")
	assert y + x == phys.Measurement(149597870700 * 2, "m**1")

# Tests whether units and arrays are correctly handled.
def test_units_3():
	p = phys.light.PhotonObject(E=phys.Measurement(5, "J**1"), v=phys.Measurement([phys.light.c, 0, 0], "m**1 s**-1"))
	assert p.E.units == {"L": 2, "T": -2, "M": 1}
	assert p.v.units == {"L": 1, "T": -1}
	assert lin.norm(p.v) == phys.light.c

# Tests whether underlying units are correct during a conversion.
def test_units_4():
	E = phys.light.E_from_wavelength(phys.Measurement(633e-9, "m**1"))
	assert E == (299792458 * 6.62607015e-34) / (633e-9)
	assert E.units == {"L": 2, "T": -2, "M": 1}
	wv = phys.light.wavelength_from_E(E)
	assert wv == 633e-9
	assert dict_equiv(wv.units, {"L": 1})

# Tests whether a conversion is correctly handled.
def test_units_5():
	E_g = phys.Measurement(0, "J**1") + phys.Measurement(13.6, "eV**1")
	f = E_g / phys.light.h
	l = phys.light.c / f
	assert E_g == 1.602176634e-19 * 13.6
	assert dict_equiv(E_g.units, {"L": 2, "T": -2, "M": 1})
	assert f == (1.602176634e-19 * 13.6) / 6.62607015e-34
	assert dict_equiv(f.units, {"T": -1})
	assert l == 299792458 / ((1.602176634e-19 * 13.6) / 6.62607015e-34)
	assert dict_equiv(l.units, {"L": 1})

# Tests whether basic ufuncs are correct.
def test_units_6():
	a = phys.Measurement(5, "kg**1 m**1 s**-2")
	l = phys.Measurement(5, "au**1")
	t = phys.Measurement(10, "min**2")
	assert a * t == 50
	assert phys.Measurement(0, "kg**1 m**1") + (a * t) == (60 ** 2) * 10 * 5
	assert a * l == 25
	assert (a / l).flat[0] == 5 / (5 * 149597870700)
	assert a ** 2 == 25
	assert dict_equiv((a ** 2).units, {"M": 2, "L": 2, "S": -4})
	assert np.sqrt(l) == np.sqrt(5)
	assert phys.Measurement(0, "m**1") + np.sqrt(l) == np.sqrt(149597870700 * 5)