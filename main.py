import os
from typing import List, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="DeepContent AI", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://deepcontent-frontend.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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


class CalendarResponse(BaseModel):
    days: List[DayItem]


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
            value_logic="Aide l'audience à comprendre pourquoi le sujet compte.",
        ),
        Pillar(
            name="Preuves",
            description="Cas concrets, résultats, démonstrations.",
            value_logic="Rassure et crédibilise l'expertise.",
        ),
        Pillar(
            name="Méthodes",
            description="Frameworks, étapes, routines et checklists.",
            value_logic="Donne quelque chose de pratique à appliquer.",
        ),
    ]

    system_prompt = """
Tu es un stratège éditorial senior francophone.
Tu génères un calendrier éditorial de 30 jours vraiment utile, crédible et varié.

Règles obligatoires :
- Réponds en français.
- Génère exactement 30 idées.
- Chaque idée doit être distincte, spécifique et non répétitive.
- Interdiction de recycler la même structure de phrase dans tous les titres.
- Varie fortement les angles : opinion, tutoriel, erreur, étude de cas, objection, coulisses, preuve, storytelling, analyse, comparaison, checklist, FAQ.
- Les topics doivent être concrets et plausibles pour la cible.
- Pas de titres génériques du type "Sujet stratégique jour X".
- Le contenu doit aider la personne à atteindre son objectif business.
- Répartis intelligemment les idées entre les piliers Fondations, Preuves et Méthodes.
- Répartis les formats entre Post LinkedIn, Carousel et Newsletter.
- Répartis les content_type entre educatif, preuve, opinion, coulisses et promo.
- Utilise deep_dive seulement pour quelques jours, pas tous.
- Mets evergreen à true seulement quand l'idée reste valable longtemps.
- status doit toujours être "idea".
"""

    user_prompt = f"""
Crée un calendrier éditorial de 30 jours pour :

Activité : {payload.activity}
Audience : {payload.audience}
Objectif principal : {payload.main_goal}
Plateforme : {payload.platform}
Ton : {payload.tone}
Angle souhaité : {payload.angle or "non précisé"}

Piliers disponibles :
{[p.model_dump() for p in pillars]}

Retourne uniquement les 30 objets day dans le format demandé.
"""

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=CalendarResponse,
    )

    parsed = response.output_parsed

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
        "days": [d.model_dump() for d in parsed.days],
    }
