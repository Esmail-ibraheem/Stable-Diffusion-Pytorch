from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sd import generate_image
from PIL import Image
from io import BytesIO
import base64
from sqlalchemy import create_engine, Column, String, Integer, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI()

DATABASE_URL = "sqlite:///./projects.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    hardware = Column(String)

Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")

class PromptRequest(BaseModel):
    prompt: str

class ProjectRequest(BaseModel):
    name: str
    description: str
    hardware: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/create_project/")
async def create_project(project_request: ProjectRequest):
    db = SessionLocal()
    project = db.query(Project).filter(Project.name == project_request.name).first()
    if project:
        db.close()
        raise HTTPException(status_code=400, detail="Project with this name already exists.")
    
    new_project = Project(
        name=project_request.name,
        description=project_request.description,
        hardware=project_request.hardware
    )
    db.add(new_project)
    db.commit()
    db.close()
    return {"message": "Project created successfully"}

@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    try:
        output_image = generate_image(prompt)
        img = Image.fromarray(output_image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
