import time


def retry(max_attempts=3, delay=1):

    def decorator(func):

        def wrapper(*args, **kwargs):

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)

        return wrapper

    return decorator
