import asyncio
import inspect


def run(fn, *args, **kwargs):
    async def _inner():
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return asyncio.run(_inner())
