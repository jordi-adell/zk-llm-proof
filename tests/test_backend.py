from zkllms import backend


def test_run_returns_value_from_a_sync_function():
    assert backend.run(lambda a, b: a + b, 2, 3) == 5


def test_run_awaits_an_awaitable_result():
    async def _coro(value):
        return value * 2

    assert backend.run(_coro, 21) == 42
