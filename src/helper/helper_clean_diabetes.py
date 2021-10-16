from pandas import Series
from typing import List
import re


def is_cat(series: Series):
    return series.dtype.name == 'category' or series.dtype.name == 'int8'


def transform_cat(series: Series, category_names: List[str]):
    if not is_cat(series):
        raise TypeError(
            f'Input series must be of type "category".\nParse series.astype("category")'
        )
    return series.cat.reorder_categories(category_names, ordered=True).cat.codes


def fix_age(age: Series):
    """
    Function that takes a pandas series transforms age when
    given in days into age in years.
    """
    transformed = age.copy()
    mask = age > 200
    transformed[mask] = age[mask] / 365
    return transformed.astype('float')


def fix_gender(gender: Series):
    def is_female(input_string: str):
        if input_string == "f" or input_string == "female":
            return 1
        return 0

    transformed = gender.copy()
    return transformed.apply(is_female)


def fix_pressure(pressure: Series):
    pressure_high = pressure.apply(lambda x: float(re.findall('[0-9]+', x)[0]))
    pressure_low = pressure.apply(lambda x: float(re.findall('[0-9]+', x)[1]))
    return pressure_high, pressure_low
