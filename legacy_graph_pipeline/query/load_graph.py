"""
Load our own validated IFC extraction (from extract_ifc_graph.py) into Neo4j.

Requires: pip install neo4j python-dotenv
Requires: a running Neo4j instance (docker compose up -d, from project root).

Usage:
    python3 load_graph.py <extracted_graph.json> [--clear]
"""

import sys
import json
import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load .env from the project root (two levels up from scripts/query/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

if not NEO4J_USER or not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_USER / NEO4J_PASSWORD not found. Check that a .env file exists "
        "at the project root with both set, matching what docker-compose.yml used "
        "to start the container."
    )

SCHEMA_CYPHER = [
    "CREATE CONSTRAINT space_guid IF NOT EXISTS FOR (s:Space) REQUIRE s.guid IS UNIQUE",
    "CREATE CONSTRAINT door_guid IF NOT EXISTS FOR (d:Door) REQUIRE d.guid IS UNIQUE",
    "CREATE CONSTRAINT window_guid IF NOT EXISTS FOR (w:Window) REQUIRE w.guid IS UNIQUE",
    "CREATE CONSTRAINT storey_name IF NOT EXISTS FOR (st:Storey) REQUIRE st.name IS UNIQUE",
    "CREATE CONSTRAINT equipment_guid IF NOT EXISTS FOR (e:Equipment) REQUIRE e.guid IS UNIQUE",
]


def load_graph(driver, building, clear=False):
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")

        for stmt in SCHEMA_CYPHER:
            session.run(stmt)

        for storey in building["storeys"]:
            session.run(
                "MERGE (st:Storey {name: $name}) SET st.guid = $guid",
                name=storey["name"], guid=storey["guid"],
            )

        for s in building["spaces"]:
            session.run(
                """
                MERGE (sp:Space {guid: $guid})
                SET sp.number = $number,
                    sp.function = $function,
                    sp.function_category = $function_category,
                    sp.area_m2 = $area_m2
                """,
                guid=s["guid"], number=s["number"], function=s["function"],
                function_category=s.get("function_category"), area_m2=s["area_m2"],
            )
            if s.get("storey"):
                session.run(
                    """
                    MATCH (sp:Space {guid: $guid})
                    MATCH (st:Storey {name: $storey})
                    MERGE (sp)-[:ON_STOREY]->(st)
                    """,
                    guid=s["guid"], storey=s["storey"],
                )

        for d in building["doors"]:
            session.run(
                "MERGE (dr:Door {guid: $guid}) SET dr.name = $name",
                guid=d["guid"], name=d["name"],
            )
            if d.get("storey"):
                session.run(
                    """
                    MATCH (dr:Door {guid: $guid})
                    MATCH (st:Storey {name: $storey})
                    MERGE (dr)-[:ON_STOREY]->(st)
                    """,
                    guid=d["guid"], storey=d["storey"],
                )
            for space_guid in d.get("bounds_spaces", []):
                session.run(
                    """
                    MATCH (dr:Door {guid: $door_guid})
                    MATCH (sp:Space {guid: $space_guid})
                    MERGE (dr)-[:BOUNDS]->(sp)
                    """,
                    door_guid=d["guid"], space_guid=space_guid,
                )

        for w in building["windows"]:
            session.run(
                "MERGE (win:Window {guid: $guid}) SET win.name = $name",
                guid=w["guid"], name=w["name"],
            )
            for space_guid in w.get("bounds_spaces", []):
                session.run(
                    """
                    MATCH (win:Window {guid: $win_guid})
                    MATCH (sp:Space {guid: $space_guid})
                    MERGE (win)-[:BOUNDS]->(sp)
                    """,
                    win_guid=w["guid"], space_guid=space_guid,
                )

        # building.get() -- older extracted JSON files predate this key and
        # won't have it; load everything else and just skip equipment.
        for e in building.get("equipment", []):
            session.run(
                """
                MERGE (eq:Equipment {guid: $guid})
                SET eq.name = $name,
                    eq.raw_name = $raw_name,
                    eq.ifc_type = $ifc_type,
                    eq.object_type = $object_type
                """,
                guid=e["guid"], name=e["name"], raw_name=e.get("raw_name"),
                ifc_type=e.get("ifc_type"), object_type=e.get("object_type"),
            )
            if e.get("storey"):
                session.run(
                    """
                    MATCH (eq:Equipment {guid: $guid})
                    MATCH (st:Storey {name: $storey})
                    MERGE (eq)-[:ON_STOREY]->(st)
                    """,
                    guid=e["guid"], storey=e["storey"],
                )
            if e.get("located_in_space"):
                session.run(
                    """
                    MATCH (eq:Equipment {guid: $eq_guid})
                    MATCH (sp:Space {guid: $space_guid})
                    MERGE (eq)-[:LOCATED_IN]->(sp)
                    """,
                    eq_guid=e["guid"], space_guid=e["located_in_space"],
                )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 load_graph.py <extracted_graph.json> [--clear]")
        sys.exit(1)

    path = sys.argv[1]
    clear = "--clear" in sys.argv

    with open(path) as f:
        building = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    load_graph(driver, building, clear=clear)
    driver.close()

    print(f"Loaded {len(building['spaces'])} spaces, {len(building['doors'])} doors, "
          f"{len(building['windows'])} windows, {len(building['storeys'])} storeys, "
          f"{len(building.get('equipment', []))} equipment items into Neo4j.")
