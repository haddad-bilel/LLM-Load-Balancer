from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from pipeline import run_pipeline


app = FastAPI(title="Requirement Extraction API")


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.post("/process")
async def process_documents(
    spec_file: UploadFile = File(..., description="Exactly one LLM specification file"),
    other_files: Optional[List[UploadFile]] = File(None, description="Zero or more related documents"),
):
    # Validate single spec file present
    if spec_file is None:
        raise HTTPException(status_code=400, detail="spec_file is required")

    # Create a temp workspace; ensure cleanup after run
    with tempfile.TemporaryDirectory(prefix="reqext_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Save spec file with a prefix for traceability
        spec_name = spec_file.filename or "spec"
        spec_path = tmp_path / f"SPEC__{spec_name}"
        spec_bytes = await spec_file.read()
        spec_path.write_bytes(spec_bytes)

        # Save other files if any
        saved_others: List[str] = []
        if other_files:
            for f in other_files:
                if f is None:
                    continue
                name = f.filename or "file"
                # Avoid name collision with spec
                dst = tmp_path / name
                # If file exists, add numeric suffix
                base = dst.stem
                ext = dst.suffix
                counter = 1
                while dst.exists():
                    dst = tmp_path / f"{base}_{counter}{ext}"
                    counter += 1
                content = await f.read()
                dst.write_bytes(content)
                saved_others.append(dst.name)

        # Run pipeline on the temp directory
        result = await run_pipeline(tmp_path)

        # Compose response with pipeline result plus meta
        payload = {
            "inputs": {
                "spec_file": spec_path.name,
                "other_files": saved_others,
            },
            "result": result.to_json(),  # already pretty JSON string
        }
        # Return parsed JSON for better client ergonomics
        return JSONResponse(content={"inputs": payload["inputs"], **( __import__("json").loads(result.to_json()) )})


if __name__ == "__main__":
    # For local dev:
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)


