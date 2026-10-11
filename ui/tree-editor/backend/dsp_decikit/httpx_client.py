from httpx import AsyncClient, AsyncHTTPTransport


def httpx_client_factory(**kwargs) -> AsyncClient:

    default_config = {
        "transport": AsyncHTTPTransport(retries=3),
        "timeout": 30.0,
    }

    return AsyncClient(**(default_config | kwargs))
