"""What the server says when it cannot do what was asked.

Every failure leaves through here, and every one of them comes back in the same shape::

    {"error": "busy", "message": "..."}

``error`` is a short stable code a client can branch on; ``message`` is the sentence a human
reads, and it is expected to say what to change. FastAPI's own validation errors are reshaped
into the same envelope so a caller never has to parse two different error formats.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kova_tts.paths import MissingArtifact
from kova_tts.server.protocol import ErrorResponse

log = logging.getLogger(__name__)

#: HTTP status used when a generation is already running. 409 rather than 503: nothing is
#: broken and nothing is overloaded -- the request conflicts with the one in flight, and
#: retrying it once that one finishes will succeed.
BUSY_STATUS = 409


class Busy(RuntimeError):
    """A generation is already running and this request will not wait any longer for it."""


class InvalidRequest(ValueError):
    """The request parsed, but something in it cannot be honoured (bad sampling, say)."""


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=code, message=message)
    return JSONResponse(status_code=status, content=body.model_dump())


def _flatten(exc: RequestValidationError) -> str:
    """Pydantic's error list as one sentence, keeping the field names it complained about.

    The default FastAPI body is a list of dicts with ``loc`` tuples, which is precise and
    unreadable. Callers of a local server are usually a person with curl, so the field path is
    joined back into dotted form and the messages are separated by semicolons.
    """
    parts: list[str] = []
    for error in exc.errors():
        # loc[0] is always "body"/"query"; naming it adds nothing for a JSON body.
        location = ".".join(str(item) for item in error.get("loc", ())[1:])
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "the request body could not be parsed"


def install(app: FastAPI) -> FastAPI:
    """Attach the handlers that turn engine exceptions into the error envelope."""

    @app.exception_handler(Busy)
    async def _busy(_request: Request, exc: Busy) -> JSONResponse:
        return _envelope(BUSY_STATUS, "busy", str(exc))

    @app.exception_handler(MissingArtifact)
    async def _missing(_request: Request, exc: MissingArtifact) -> JSONResponse:
        # A voice name that does not resolve, or a model file that moved: both are "you asked
        # for something that is not on this machine", which is a 404 and not a server fault.
        return _envelope(404, "not_found", str(exc))

    @app.exception_handler(InvalidRequest)
    async def _invalid(_request: Request, exc: InvalidRequest) -> JSONResponse:
        return _envelope(422, "invalid_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(422, "invalid_request", _flatten(exc))

    return app
