from typing import List, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="DeepContent AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    activity: str = Field(..., examples=["consultant B2B"])
    audience: str = Field(..., examples=["founders SaaS"])
    main_goal: str = Field(..., examples=["signer 3 clients"])
    platform: str = Field(..., examples=["LinkedIn"])
    tone: str = Field(..., examples=["direct, clair, pro"])
    angle: str = Field("", examples=["contenu de fond et preuve"])

class Pillar(BaseModel):
    name: str
    description: str
    value_logic: str

class DayItem(BaseModel):
    day: int
    pillar: str
    topic: str
    angle: str
    format: str
    content_type: str
    depth: Literal["standard", "deep_dive"]
    evergreen: bool
    status: str = "idea"

@app.get("/")
async def root():
    return {"message": "DeepContent AI backend is running"}

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/generate-calendar")
async def generate_calendar(payload: GenerateRequest):
    pillars = [
        Pillar(
            name="Fondations",
            description="Clarifier le problème, le contexte et les bases.",
            value_logic="Aide l'audience à comprendre pourquoi le sujet compte."
        ),
        Pillar(
            name="Preuves",
            description="Cas concrets, résultats, démonstrations.",
            value_logic="Rassure et crédibilise l'expertise."
        ),
        Pillar(
            name="Méthodes",
            description="Frameworks, étapes, routines et checklists.",
            value_logic="Donne quelque chose de pratique à appliquer."
        ),
    ]

    days: List[DayItem] = []
    content_types = ["educatif", "preuve", "opinion", "coulisses", "promo"]
    formats = ["Post LinkedIn", "Carousel", "Post LinkedIn", "Post LinkedIn", "Newsletter"]

    for i in range(1, 31):
        pillar = pillars[(i - 1) % len(pillars)].name
        ctype = content_types[(i - 1) % len(content_types)]
        fmt = formats[(i - 1) % len(formats)]
        depth = "deep_dive" if i in [5, 12, 19, 26] else "standard"
        evergreen = i in [2, 7, 14, 21, 28]

        days.append(
            DayItem(
                day=i,
                pillar=pillar,
                topic=f"Sujet stratégique jour {i} pour {payload.activity}",
                angle=f"Angle orienté {payload.audience} : {payload.angle or payload.main_goal}",
                format=fmt,
                content_type=ctype,
                depth=depth,
                evergreen=evergreen,
                status="idea",
            )
        )

    return {
        "meta": {
            "activity": payload.activity,
            "audience": payload.audience,
            "main_goal": payload.main_goal,
            "platform": payload.platform,
            "tone": payload.tone,
            "angle": payload.angle,
            "pillars": [p.model_dump() for p in pillars],
        },
        "days": [d.model_dump() for d in days],
    }
    import os
    if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
