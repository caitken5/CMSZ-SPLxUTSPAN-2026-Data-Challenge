"""Data adapter module for physics simulations."""

from physics.constants import FEET_TO_METERS, METERS_TO_FEET, INCHES_TO_METERS, METERS_TO_INCHES

def ft_to_m(x):
    return x * FEET_TO_METERS

def m_to_ft(x):
    return x * METERS_TO_FEET

def inches_to_meters(x):
    return x * INCHES_TO_METERS

def meters_to_inches(x):
    return x * METERS_TO_INCHES