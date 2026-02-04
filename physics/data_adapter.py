"""Data adapter module for physics simulations."""

from physics.constants import FEET_TO_METERS


def parse_array_json(s):
    # Replace bare nan with null (JSON compatible)
    s = s.replace('nan', 'null')
    return np.array(json.loads(s), dtype=np.float32)

for col in keypoint_names:
    df[col] = df[col].apply(parse_array_json)

def ft_to_m(x):
    return x * FEET_TO_METERS

def m_to_ft(x):
    return x * METERS_TO_FEET

def inches_to_meters(x):
    return x * INCHES_TO_METERS

def meters_to_inches(x):
    return x * METERS_TO_INCHES