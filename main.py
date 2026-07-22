import os
from datetime import datetime, timezone
from typing import Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
            description="Clarifier le problÃ¨me, le contexte et les bases.",
            value_logic="Aide l'audience Ã  comprendre pourquoi le sujet compte.",
        ),
        Pillar(
            name="Preuves",
            description="Cas concrets, rÃ©sultats, dÃ©monstrations.",
            value_logic="Rassure et crÃ©dibilise l'expertise.",
        ),
        Pillar(
            name="MÃ©thodes",
            description="Frameworks, Ã©tapes, routines et checklists.",
            value_logic="Donne quelque chose de pratique Ã  appliquer.",
        ),
    ]

    now_utc = datetime.now(timezone.utc)
    current_date_fr = now_utc.strftime("%d/%m/%Y")
    current_year = now_utc.year

    system_prompt = f"""
Tu es un stratÃ¨ge Ã©ditorial B2B francophone spÃ©cialisÃ© en contenu LinkedIn utile, crÃ©dible et orientÃ© business.

Contexte temporel obligatoire :
- Date actuelle : {current_date_fr}
- AnnÃ©e actuelle : {current_year}
- Toute recommandation doit Ãªtre cohÃ©rente avec cette date.
- N'utilise jamais 2024 ni 2025 dans les titres, angles ou idÃ©es, sauf si l'utilisateur demande explicitement une rÃ©trospective historique.
- Par dÃ©faut, n'inclus pas d'annÃ©e dans les titres.
- Si une rÃ©fÃ©rence temporelle est vraiment utile, elle doit Ãªtre cohÃ©rente avec {current_year}.
- Ã‰vite toute formulation datÃ©e, obsolÃ¨te ou figÃ©e dans une annÃ©e passÃ©e.

Ta mission :
gÃ©nÃ©rer un calendrier Ã©ditorial de 30 jours en franÃ§ais, Ã  partir des informations fournies par lâ€™utilisateur.

Le calendrier doit Ãªtre pensÃ© pour une personne qui vend une expertise, un service, un accompagnement ou du conseil Ã  une audience francophone.
Le contenu doit aider Ã  :
- construire la crÃ©dibilitÃ©,
- clarifier le positionnement,
- crÃ©er de la confiance,
- prÃ©parer la conversion,
- donner de vraies idÃ©es publiables.

Contexte utilisateur :
- ActivitÃ© : {payload.activity}
- Audience : {payload.audience}
- Objectif principal : {payload.main_goal}
- Plateforme : {payload.platform}
- Ton : {payload.tone}
- Angle : {payload.angle or "non prÃ©cisÃ©"}

Consignes gÃ©nÃ©rales :
- RÃ©ponds uniquement en franÃ§ais.
- Ne produis aucun texte hors JSON.
- Ne mets aucun markdown.
- Ne mets aucune explication.
- Le rÃ©sultat doit Ãªtre directement exploitable dans une application.
- Le niveau doit Ãªtre concret, crÃ©dible, prÃ©cis et publiable.
- Ã‰vite les sujets vagues, creux, trop gÃ©nÃ©riques ou interchangeables.
- Ã‰vite les banalitÃ©s du type "lâ€™importance de...", "comment rÃ©ussir...", "les clÃ©s de...".
- PrÃ©fÃ¨re des angles tirÃ©s du rÃ©el : objections clients, erreurs frÃ©quentes, signaux de confiance, preuves, coulisses, analyses terrain, mÃ©thodes concrÃ¨tes, avant/aprÃ¨s, dÃ©cisions business, erreurs observÃ©es.
- Le calendrier doit sembler conÃ§u pour une vraie activitÃ©, pas pour un template gÃ©nÃ©rique.
- Les idÃ©es doivent Ãªtre adaptÃ©es Ã  lâ€™audience et Ã  lâ€™objectif business.
- Le ton doit rester cohÃ©rent avec les informations fournies.
- Varie les angles, les rythmes et les intentions.
- Ne rÃ©pÃ¨te pas plusieurs fois la mÃªme idÃ©e avec un titre lÃ©gÃ¨rement diffÃ©rent.
- Les titres doivent Ãªtre spÃ©cifiques, utiles et assez forts pour donner envie de publier.
- Le contenu doit Ãªtre pensÃ© pour la conversion indirecte : visibilitÃ©, crÃ©dibilitÃ© ou conversion.
- Le calendrier doit Ãªtre meilleur quâ€™un calendrier IA gÃ©nÃ©rique.
- Les angles doivent reflÃ©ter des enjeux actuels et crÃ©dibles pour {current_year}.
- Aucun titre ne doit sembler Ã©crit pour 2024 ou 2025.

Structure attendue :
Tu dois retourner un objet JSON avec une clÃ© :
- "days"

Contraintes sur les 30 jours :
- GÃ©nÃ¨re exactement 30 objets dans "days"
- day : entier de 1 Ã  30
- pillar : "Fondations", "Preuves" ou "MÃ©thodes"
- rÃ©partis les 30 jours de faÃ§on Ã©quilibrÃ©e entre les 3 piliers
- topic : titre en franÃ§ais, spÃ©cifique, crÃ©dible, non gÃ©nÃ©rique
- angle : angle Ã©ditorial court en franÃ§ais, parmi des approches comme opinion, analyse, objection, Ã©tude de cas, checklist, coulisses, preuve, tutoriel, comparaison, storytelling, FAQ, audit, framework
- format : choisis parmi "Post LinkedIn", "Carousel", "Newsletter"
- content_type : choisis parmi "educatif", "preuve", "opinion", "coulisses", "promo"
- depth : choisis parmi "standard" ou "deep_dive"
- evergreen : boolÃ©en
- goal : choisis parmi "visibilite", "credibilite", "conversion"
- status : toujours "idea"

Contraintes de qualitÃ© :
- Au moins 8 idÃ©es doivent Ãªtre trÃ¨s orientÃ©es crÃ©dibilitÃ© ou preuve.
- Au moins 6 idÃ©es doivent partir dâ€™objections, dâ€™erreurs ou de problÃ¨mes frÃ©quents.
- Au moins 6 idÃ©es doivent Ãªtre immÃ©diatement actionnables par lâ€™audience.
- Au moins 4 idÃ©es doivent pouvoir soutenir indirectement une vente ou une prise de contact.
- Il faut un bon Ã©quilibre entre visibilitÃ©, crÃ©dibilitÃ© et conversion indirecte.
- Les sujets doivent donner lâ€™impression quâ€™ils viennent dâ€™une vraie pratique terrain.
- Les titres doivent Ãªtre plus prÃ©cis que des banalitÃ©s de consultant.
- Pas de jargon inutile.
- Pas de titres trop abstraits.
- Pas de rÃ©pÃ©titions dÃ©guisÃ©es.
- Pas de sujet hors cible.
- Pas dâ€™anglais, sauf nom de plateforme ou termes impossibles Ã  traduire naturellement.
- Ne fais aucune rÃ©fÃ©rence obsolÃ¨te Ã  une annÃ©e passÃ©e.
- Si une annÃ©e apparaÃ®t, elle doit Ãªtre {current_year}.

Logique des piliers :
- Fondations : clarifier le problÃ¨me, le positionnement, le point de vue, les erreurs de perception, les croyances du marchÃ©
- Preuves : rassurer, montrer le rÃ©el, prouver lâ€™expÃ©rience, traiter les objections, montrer des signaux crÃ©dibles
- MÃ©thodes : transmettre des frameworks, checklists, routines, faÃ§ons de faire, Ã©tapes concrÃ¨tes, modÃ¨les rÃ©utilisables

Logique du champ goal :
- visibilite : contenu conÃ§u pour attirer lâ€™attention des bonnes personnes
- credibilite : contenu conÃ§u pour rassurer et dÃ©montrer la compÃ©tence
- conversion : contenu conÃ§u pour rapprocher un prospect dâ€™un Ã©change, dâ€™un message ou dâ€™une prise de contact

RÃ¨gles de style :
- Ã‰cris comme un stratÃ¨ge Ã©ditorial intelligent, sobre, concret.
- Pas de sensationnalisme.
- Pas de promesses exagÃ©rÃ©es.
- Pas de formulations creuses.
- On veut des idÃ©es qui donnent envie dâ€™Ã©crire un vrai post utile, pas des titres de contenu IA gÃ©nÃ©rique.
- Chaque idÃ©e doit avoir un intÃ©rÃªt clair pour lâ€™audience.
- Si possible, fais sentir la tension business, la rÃ©alitÃ© du terrain ou la psychologie du prospect.

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
        if isinstance(day_data.get("topic"), str):
            day_data["topic"] = day_data["topic"].replace("2024", str(current_year)).replace("2025", str(current_year))
        if isinstance(day_data.get("angle"), str):
            day_data["angle"] = day_data["angle"].replace("2024", str(current_year)).replace("2025", str(current_year))
        cleaned_days.append(day_data)

    return {
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
