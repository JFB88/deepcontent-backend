import os
from typing import List, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = FastAPI(title="DeepContent AI", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


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

    system_prompt = f"""
Tu es un stratège éditorial B2B francophone spécialisé en contenu LinkedIn utile, crédible et orienté business.

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

Exemple de bon niveau de précision :
- "Pourquoi la plupart des consultants B2B publient beaucoup mais signent peu"
- "L’objection 'LinkedIn ne m’apporte rien' cache souvent 3 problèmes très précis"
- "Le type de preuve le plus sous-utilisé par les indépendants B2B"
- "Ce qu’on comprend sur un marché quand on relit 20 appels de découverte"

Exemple de mauvais niveau :
- "L’importance du personal branding"
- "Comment réussir sur LinkedIn"
- "Pourquoi le contenu est important"
- "Les clés d’une bonne stratégie"

Retourne uniquement le JSON final.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=system_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CalendarResponse,
            temperature=0.9,
        ),
    )

    parsed = response.parsed

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
