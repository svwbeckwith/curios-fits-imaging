import math
from datetime import datetime


def radec_to_altaz_approx(ra_hours: float, dec_deg: float, dateobs: str,
                          latitude_deg: float = 37.78111,
                          longitude_deg: float = -122.39139):
    """Approximate RA/Dec to Alt/Az conversion copied from the lab notebook.

    This preserves the notebook convention and should be replaced by Astropy for
    precision pointing work.
    """
    date_string = dateobs[:10]
    time_string = dateobs[11:]
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    day_of_year = int(date_object.strftime("%j"))
    h, m, s = map(int, time_string.split(":"))

    dst = -1 if 66 < day_of_year < 305 else 0
    fractional_hour = h + m / 60.0 + s / 3600.0
    long_frac_part, _ = math.modf(longitude_deg / 15)
    lst = fractional_hour + dst + (day_of_year - 264) * 24 / 365.24 + long_frac_part
    lst %= 24

    hour_angle_deg = 15 * (lst - ra_hours)
    ha_rad = math.radians(hour_angle_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(latitude_deg)

    sin_alt = math.sin(dec_rad) * math.sin(lat_rad) + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad)
    altitude_rad = math.asin(sin_alt)

    y = -math.sin(ha_rad) * math.cos(dec_rad)
    x = math.cos(dec_rad) * math.cos(ha_rad) * math.sin(lat_rad) - math.sin(dec_rad) * math.cos(lat_rad)
    azimuth_deg = (math.degrees(math.atan2(y, x)) + 180.0) % 360.0
    return math.degrees(altitude_rad), azimuth_deg


def parallactic_pa_from_altaz(alt_deg: float, az_deg: float, latitude_deg: float = 37.78111) -> float:
    alt = math.radians(alt_deg)
    az = math.radians(az_deg)
    tlat = math.tan(math.radians(latitude_deg))
    return math.degrees(math.atan2(math.sin(az), tlat * math.cos(alt) - math.sin(alt) * math.cos(az)))
