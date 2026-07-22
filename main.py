import os
from datetime import datetime, timezone
from typing import Dict, List, Literal
import unicodedata

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = FastAPI(title="DeepContent AI", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DAILY_LIMIT = 4
usage_store: Dict[str, Dict[str, int | str]] = {}


class GenerateRequest(BaseModel):
    activity: str = Field(..., examples=["consultant B2B"])
    audience: str = Field(..., examples=["fondateurs SaaS"])
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
    goal: Literal["visibilite", "credibilite", "conversion"]
    status: str = "idea"


class CalendarResponse(BaseModel):
    days: List[DayItem]


@app.get("/")
async def root():
    return {"message": "DeepContent AI backend is running"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def get_today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_remaining_generations(ip: str) -> int:
    today = get_today_key()
    current = usage_store.get(ip)

    if not current or current.get("date") != today:
        usage_store[ip] = {"date": today, "count": 0}
        return DAILY_LIMIT

    return max(0, DAILY_LIMIT - int(current["count"]))


def consume_generation(ip: str) -> int:
    today = get_today_key()
    current = usage_store.get(ip)

    if not current or current.get("date") != today:
        usage_store[ip] = {"date": today, "count": 0}

    if int(usage_store[ip]["count"]) >= DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Limite quotidienne atteinte. Reviens demain.",
                "remaining": 0,
                "daily_limit": DAILY_LIMIT,
            },
        )

    usage_store[ip]["count"] = int(usage_store[ip]["count"]) + 1
    return max(0, DAILY_LIMIT - int(usage_store[ip]["count"]))


def normalize_text(value):
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


@app.post("/api/generate-calendar")
async def generate_calendar(payload: GenerateRequest, request: Request):
    client_ip = get_client_ip(request)
    remaining_before = get_remaining_generations(client_ip)

    if remaining_before <= 0:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Limite quotidienne atteinte. Reviens demain.",
                "remaining": 0,
                "daily_limit": DAILY_LIMIT,
            },
        )

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

    now_utc = datetime.now(timezone.utc)
    current_date_fr = now_utc.strftime("%d/%m/%Y")
    current_year = now_utc.year

    system_prompt = f"""
Tu es un stratège éditorial B2B francophone spécialisé en contenu LinkedIn utile, crédible et orienté business.

Contexte temporel obligatoire :
- Date actuelle : {current_date_fr}
- Année actuelle : {current_year}
- Toute recommandation doit être cohérente avec cette date.
- N'utilise jamais 2024 ni 2025 dans les titres, angles ou idées, sauf si l'utilisateur demande explicitement une rétrospective historique.
- Par défaut, n'inclus pas d'année dans les titres.
- Si une référence temporelle est vraiment utile, elle doit être cohérente avec {current_year}.
- Évite toute formulation datée, obsolète ou figée dans une année passée.

Ta mission :
générer un calendrier éditorial de 30 jours en français, à partir des informations fournies par l’utilisateur.

Le calendrier doit être pensé pour une personne qui vend une expertise, un service, un accompagnement ou du conseil à une audience francophone.
Le contenu doit aider à :
- construire la crédibilité,
- clarifier le positionnement,
- créer de la confiance,
- préparer la conversion,
- donner de vraies idées publiables.

Contexte utilisateur :
- Activité : {payload.activity}
- Audience : {payload.audience}
- Objectif principal : {payload.main_goal}
- Plateforme : {payload.platform}
- Ton : {payload.tone}
- Angle : {payload.angle or "non précisé"}

Consignes générales :
- Réponds uniquement en français.
- Ne produis aucun texte hors JSON.
- Ne mets aucun markdown.
- Ne mets aucune explication.
- Le résultat doit être directement exploitable dans une application.
- Le niveau doit être concret, crédible, précis et publiable.
- Évite les sujets vagues, creux, trop génériques ou interchangeables.
- Évite les banalités du type "l’importance de...", "comment réussir...", "les clés de...".
- Préfère des angles tirés du réel : objections clients, erreurs fréquentes, signaux de confiance, preuves, coulisses, analyses terrain, méthodes concrètes, avant/après, décisions business, erreurs observées.
- Le calendrier doit sembler conçu pour une vraie activité, pas pour un template générique.
- Les idées doivent être adaptées à l’audience et à l’objectif business.
- Le ton doit rester cohérent avec les informations fournies.
- Varie les angles, les rythmes et les intentions.
- Ne répète pas plusieurs fois la même idée avec un titre légèrement différent.
- Les titres doivent être spécifiques, utiles et assez forts pour donner envie de publier.
- Le contenu doit être pensé pour la conversion indirecte : visibilité, crédibilité ou conversion.
- Le calendrier doit être meilleur qu’un calendrier IA générique.
- Les angles doivent refléter des enjeux actuels et crédibles pour {current_year}.
- Aucun titre ne doit sembler écrit pour 2024 ou 2025.

Structure attendue :
Tu dois retourner un objet JSON avec une clé :
- "days"

Contraintes sur les 30 jours :
- Génère exactement 30 objets dans "days"
- day : entier de 1 à 30
- pillar : "Fondations", "Preuves" ou "Méthodes"
- répartis les 30 jours de façon équilibrée entre les 3 piliers
- topic : titre en français, spécifique, crédible, non générique
- angle : angle éditorial court en français, parmi des approches comme opinion, analyse, objection, étude de cas, checklist, coulisses, preuve, tutoriel, comparaison, storytelling, FAQ, audit, framework
- format : choisis parmi "Post LinkedIn", "Carousel", "Newsletter"
- content_type : choisis parmi "educatif", "preuve", "opinion", "coulisses", "promo"
- depth : choisis parmi "standard" ou "deep_dive"
- evergreen : booléen
- goal : choisis parmi "visibilite", "credibilite", "conversion"
- status : toujours "idea"

Contraintes de qualité :
- Au moins 8 idées doivent être très orientées crédibilité ou preuve.
- Au moins 6 idées doivent partir d’objections, d’erreurs ou de problèmes fréquents.
- Au moins 6 idées doivent être immédiatement actionnables par l’audience.
- Au moins 4 idées doivent pouvoir soutenir indirectement une vente ou une prise de contact.
- Il faut un bon équilibre entre visibilité, crédibilité et conversion indirecte.
- Les sujets doivent donner l’impression qu’ils viennent d’une vraie pratique terrain.
- Les titres doivent être plus précis que des banalités de consultant.
- Pas de jargon inutile.
- Pas de titres trop abstraits.
- Pas de répétitions déguisées.
- Pas de sujet hors cible.
- Pas d’anglais, sauf nom de plateforme ou termes impossibles à traduire naturellement.
- Ne fais aucune référence obsolète à une année passée.
- Si une année apparaît, elle doit être {current_year}.

Logique des piliers :
- Fondations : clarifier le problème, le positionnement, le point de vue, les erreurs de perception, les croyances du marché
- Preuves : rassurer, montrer le réel, prouver l’expérience, traiter les objections, montrer des signaux crédibles
- Méthodes : transmettre des frameworks, checklists, routines, façons de faire, étapes concrètes, modèles réutilisables

Logique du champ goal :
- visibilite : contenu conçu pour attirer l’attention des bonnes personnes
- credibilite : contenu conçu pour rassurer et démontrer la compétence
- conversion : contenu conçu pour rapprocher un prospect d’un échange, d’un message ou d’une prise de contact

Règles de style :
- Écris comme un stratège éditorial intelligent, sobre, concret.
- Pas de sensationnalisme.
- Pas de promesses exagérées.
- Pas de formulations creuses.
- On veut des idées qui donnent envie d’écrire un vrai post utile, pas des titres de contenu IA générique.
- Chaque idée doit avoir un intérêt clair pour l’audience.
- Si possible, fais sentir la tension business, la réalité du terrain ou la psychologie du prospect.

Retourne uniquement le JSON final.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=system_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CalendarResponse,
            temperature=0.7,
        ),
    )

    remaining_after = consume_generation(client_ip)
    parsed = response.parsed

    cleaned_days = []
    for day in parsed.days:
        day_data = day.model_dump()
        day_data["topic"] = normalize_text(day_data.get("topic"))
        day_data["angle"] = normalize_text(day_data.get("angle"))
        if isinstance(day_data.get("topic"), str):
            day_data["topic"] = day_data["topic"].replace("2024", str(current_year)).replace("2025", str(current_year))
        if isinstance(day_data.get("angle"), str):
            day_data["angle"] = day_data["angle"].replace("2024", str(current_year)).replace("2025", str(current_year))
        cleaned_days.append(day_data)

    payload_out = {
        "meta": {
            "activity": payload.activity,
            "audience": payload.audience,
            "main_goal": payload.main_goal,
            "platform": payload.platform,
            "tone": payload.tone,
            "angle": payload.angle,
            "pillars": [p.model_dump() for p in pillars],
            "current_date": current_date_fr,
            "current_year": current_year,
        },
        "quota": {
            "daily_limit": DAILY_LIMIT,
            "remaining": remaining_after,
        },
        "days": cleaned_days,
    }

    return JSONResponse(content=payload_out, media_type="application/json; charset=utf-8")
