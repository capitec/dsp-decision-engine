from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
import os
import httpx
import uuid
from hamilton import driver
from example import main
import typing

# Assuming your node structure and types are like this:
class Node(BaseModel):
    node_id: uuid.UUID = Field(default_factory=uuid.uuid4, alias="nodeId") # TODO get id from function name hash
    name: str
    documentation: str = ""
    dependencies: typing.List[str] = []

# FastAPI app initialization
app = FastAPI()

# API router for project node retrieval
api_router = APIRouter()

@api_router.get("/projects/{project_id}/nodes")
async def get_nodes(project_id: uuid.UUID):
    # Retrieve nodes for the given project ID (here we assume the same for all for simplicity)
    if project_id == uuid.UUID("00000000-0000-0000-0000-000000000000"):
        dr = driver.Driver({}, main)
        hamilton_nodes = dr.graph.nodes
        return [Node(name=n.name, documentation=n.documentation, dependencies=[d.name for d in n.dependencies]) for n in hamilton_nodes.values()]
    else:
        raise HTTPException(status_code=404, detail="Project not found")

app.include_router(api_router,prefix="/api")


# app.mount("/", StaticFiles(directory="static"), name="static")
