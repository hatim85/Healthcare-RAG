import os
import redis
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
load_dotenv('.env.example')

@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """Return a shared, authenticated Redis Cloud Client."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )

    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT")
    username = os.getenv("REDIS_USERNAME")
    password = os.getenv("REDIS_PASSWORD")

    if not all([host, port, username, password]):
        raise RuntimeError("Redis config is incomplete. Check the REDIS_* settings.")

    try:
        port_number = int(port)
    except ValueError as exc:
        raise RuntimeError("REDIS_PORT must be a valid integer.") from exc

    if not 1 <= port_number <= 65535:
        raise RuntimeError("REDIS_PORT is outside the valid range.")

    return redis.Redis(
        host=host,
        port=port_number,
        username=username,
        password=password,
        ssl=False,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )
    
if __name__=="__main__":
    client=get_redis_client()
    try:
        if client.ping():
            print("Redis Cloud connection: SUCCESS")
            print("Redis PING: PONG")
    except redis.exceptions.RedisError as exc:
        print("Redis Cloud connection: FAILED")
        print("Error category:", type(exc).__name__)
        raise SystemExit(1)
    finally:
        client.close()