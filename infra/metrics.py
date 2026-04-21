import time


def timeit(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {round(duration, 3)}s")
        return result

    return wrapper
