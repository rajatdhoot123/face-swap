from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import os

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="No API key provided"
        )
    
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API key not configured on server"
        )
    
    if api_key_header != api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    
    return api_key_header 