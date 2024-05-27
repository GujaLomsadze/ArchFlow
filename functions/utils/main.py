import random


def fluctuate_integer(n, x_percent):
    fluctuation_range = n * x_percent / 100
    random_fluctuation = random.uniform(-fluctuation_range, fluctuation_range)
    new_value = n + int(random_fluctuation)
    return new_value
