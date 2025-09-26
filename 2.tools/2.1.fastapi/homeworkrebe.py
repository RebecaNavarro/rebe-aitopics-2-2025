# characters_api.py
import io
import csv
import random
from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

class RaceEnum(str, Enum):
    orc = "ORC"
    elf = "ELF"
    human = "HUMAN"
    goblin = "GOBLIN"

class Guild(BaseModel):
    id: int
    name: str
    realm: str
    created: datetime

class Character(BaseModel):
    id: int
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int | None = None
    guild: Guild

class CharacterCreate(BaseModel):
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int
    guild_id: int

app = FastAPI(title="Characters API")

guilds: list[Guild] = []
characters: list[Character] = []

@app.post("/guilds", status_code=201)
def create_guild(guild: Guild) -> list[Guild]:
    guilds.append(guild)
    return guilds

@app.post("/characters", status_code=201)
def create_character(character: CharacterCreate):
    id = random.randint(0, 9999)
    guilds_found = [g for g in guilds if g.id == character.guild_id]
    if not guilds_found:
        raise HTTPException(status_code=404, detail="guild not found")
    guild = guilds_found[0]
    new_character = Character(
        id=id,
        guild=guild,
        **character.model_dump(exclude={"guild_id"})
    )
    characters.append(new_character)
    return new_character

@app.get("/characters")
def get_characters() -> list[Character]:
    return characters

@app.get("/characters/report", responses={200: {"content": {"text/csv": {}}}})
def export_characters() -> Response:
    if not characters:
        return Response(content="No characters available", media_type="text/plain")

    csv_stream = io.StringIO()
    fieldnames = ["id", "name", "level", "race", "hp", "damage", "guild_id", "guild_name", "guild_realm"]
    writer = csv.DictWriter(csv_stream, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

    writer.writeheader()
    for char in characters:
        writer.writerow({
            "id": char.id,
            "name": char.name,
            "level": char.level,
            "race": char.race,
            "hp": char.hp,
            "damage": char.damage,
            "guild_id": char.guild.id,
            "guild_name": char.guild.name,
            "guild_realm": char.guild.realm,
        })

    text = csv_stream.getvalue()
    return Response(content=text, media_type="text/csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("characters_api:app", reload=True, port=8010)
