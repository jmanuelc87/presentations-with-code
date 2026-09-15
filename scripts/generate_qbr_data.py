#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de datos sintéticos para la demo de QBR (Business Review ejecutivo).

Empresa proveedora (la que hace el QBR): "Nimbus Cloud", un SaaS B2B.
Sus clientes (cuentas) son empresas ficticias clásicas. Cada cuenta tiene una
HISTORIA plantada de forma coherente entre:
  - los NÚMEROS (SQLite: ingresos, uso, salud)  -> los encuentra el analista text-to-SQL
  - el TEXTO (tickets de soporte + notas de cuenta en español) -> lo encuentra el RAG

Periodo: Q1 2026 (ene-mar) y Q2 2026 (abr-jun). El QBR es del Q2.

Reproducible: semilla fija. Vuelve a correr para regenerar idéntico.
"""

import csv
import os
import random
import sqlite3
import textwrap
from datetime import date, timedelta

from faker import Faker

SEED = 42
random.seed(SEED)
fake = Faker("es_MX")
Faker.seed(SEED)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NOTES_DIR = os.path.join(OUT, "notas_cuenta")
DOCS_DIR = os.path.join(OUT, "docs_rag")  # corpus combinado para el vector store
os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
Q1 = MONTHS[:3]
Q2 = MONTHS[3:]


def quarter_of(m):
    return "Q1-2026" if m in Q1 else "Q2-2026"


# ---------------------------------------------------------------------------
# 1) DEFINICIÓN DE CUENTAS Y SUS ARCOS NARRATIVOS
# ---------------------------------------------------------------------------
# health_score y CSAT en escala 0-100 / 1-5. mrr_base en USD/mes.
# 'arc' describe la trayectoria mensual de uso e ingresos como multiplicadores.
ACCOUNTS = [
    {
        "account_id": "ACME",
        "name": "Acme Corporation",
        "industry": "Manufactura",
        "country": "México",
        "region": "LATAM",
        "segment": "Enterprise",
        "csm_owner": "Valeria Núñez",
        "contract_start": "2023-02-01",
        "renewal_quarter": "Q3-2026",
        "seats": 450,
        "mrr_base": 38000,
        # RIESGO RUIDOSO: cae en Q2 en números Y en tickets.
        "rev_arc": [1.00, 1.00, 0.98, 0.82, 0.74, 0.70],
        "usage_arc": [1.00, 0.97, 0.95, 0.80, 0.68, 0.60],
        "story": "risk_loud",
    },
    {
        "account_id": "GLOBEX",
        "name": "Globex S.A.",
        "industry": "Retail",
        "country": "México",
        "region": "LATAM",
        "segment": "Enterprise",
        "csm_owner": "Diego Hernández",
        "contract_start": "2022-09-15",
        "renewal_quarter": "Q4-2026",
        "seats": 300,
        "mrr_base": 27000,
        # EXPANSIÓN: crece en Q2, alta adopción.
        "rev_arc": [1.00, 1.02, 1.05, 1.12, 1.20, 1.28],
        "usage_arc": [1.00, 1.04, 1.08, 1.15, 1.22, 1.30],
        "story": "expansion",
    },
    {
        "account_id": "INITECH",
        "name": "Initech",
        "industry": "Tecnología",
        "country": "Colombia",
        "region": "LATAM",
        "segment": "Mid-Market",
        "csm_owner": "Diego Hernández",
        "contract_start": "2024-01-10",
        "renewal_quarter": "Q1-2027",
        "seats": 120,
        "mrr_base": 9500,
        # FRICCIÓN: ingresos planos, suben tickets sobre una función (reportes).
        "rev_arc": [1.00, 1.00, 1.01, 1.00, 0.99, 1.00],
        "usage_arc": [1.00, 1.01, 1.00, 0.98, 0.97, 0.96],
        "story": "friction",
    },
    {
        "account_id": "UMBRELLA",
        "name": "Umbrella Corp",
        "industry": "Salud",
        "country": "España",
        "region": "EMEA",
        "segment": "Mid-Market",
        "csm_owner": "Valeria Núñez",
        "contract_start": "2026-01-05",
        "renewal_quarter": "Q1-2027",
        "seats": 80,
        "mrr_base": 6000,
        # LOGO NUEVO: arranca en Q1 y rampa, onboarding.
        "rev_arc": [0.40, 0.70, 1.00, 1.10, 1.20, 1.25],
        "usage_arc": [0.30, 0.65, 0.95, 1.10, 1.25, 1.35],
        "story": "new_logo",
    },
    {
        "account_id": "SOYLENT",
        "name": "Soylent Industries",
        "industry": "Alimentos y Bebidas",
        "country": "México",
        "region": "LATAM",
        "segment": "Enterprise",
        "csm_owner": "Diego Hernández",
        "contract_start": "2021-06-01",
        "renewal_quarter": "Q2-2027",
        "seats": 520,
        "mrr_base": 41000,
        # ESTRELLA ESTABLE: cuenta de referencia, alta CSAT.
        "rev_arc": [1.00, 1.01, 1.00, 1.01, 1.02, 1.02],
        "usage_arc": [1.00, 1.00, 1.01, 1.01, 1.02, 1.02],
        "story": "stable_star",
    },
    {
        "account_id": "HOOLI",
        "name": "Hooli",
        "industry": "Tecnología",
        "country": "Chile",
        "region": "LATAM",
        "segment": "Enterprise",
        "csm_owner": "Valeria Núñez",
        "contract_start": "2023-11-20",
        "renewal_quarter": "Q3-2026",
        "seats": 260,
        "mrr_base": 22000,
        # CHURN SILENCIOSO: ingresos planos pero uso se desploma; casi sin tickets.
        "rev_arc": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "usage_arc": [1.00, 0.92, 0.83, 0.70, 0.55, 0.42],
        "story": "silent_churn",
    },
]

# salud trimestral plantada (nps -100..100, csat 1..5, health 0..100)
HEALTH = {
    "ACME":     {"Q1-2026": (32, 4.1, 71), "Q2-2026": (-18, 2.9, 38)},
    "GLOBEX":   {"Q1-2026": (44, 4.4, 80), "Q2-2026": (58, 4.6, 88)},
    "INITECH":  {"Q1-2026": (20, 3.8, 64), "Q2-2026": (8,  3.4, 55)},
    "UMBRELLA": {"Q1-2026": (15, 3.9, 60), "Q2-2026": (40, 4.3, 78)},
    "SOYLENT":  {"Q1-2026": (62, 4.7, 90), "Q2-2026": (66, 4.7, 91)},
    "HOOLI":    {"Q1-2026": (30, 4.0, 68), "Q2-2026": (10, 3.6, 47)},
}

PRODUCTS = ["Nimbus Core", "Nimbus Analytics", "Nimbus Connect (API)"]


# ---------------------------------------------------------------------------
# 2) PLANTILLAS DE TICKETS EN ESPAÑOL (contenido real, no lorem ipsum)
# ---------------------------------------------------------------------------
TICKET_TEMPLATES = {
    "Caída del servicio": [
        ("Servicio caído en producción",
         "El panel de {prod} lleva {mins} minutos inaccesible para todo nuestro equipo. "
         "Recibimos error 503 al iniciar sesión. Esto está bloqueando la operación diaria; "
         "necesitamos una solución urgente y una explicación de la causa raíz."),
        ("Intermitencias y errores 500",
         "Desde esta mañana {prod} responde de forma intermitente con errores 500. "
         "Es la {nth} caída este mes y nuestro equipo de operaciones está muy molesto. "
         "¿Pueden confirmar si hay un incidente abierto?"),
        ("Caída durante horario crítico",
         "El servicio se cayó justo durante nuestro cierre de turno. La falta de disponibilidad "
         "nos costó varias horas de trabajo. Solicitamos un reporte post-incidente formal."),
    ],
    "Rendimiento": [
        ("Lentitud severa al cargar reportes",
         "Los reportes en {prod} tardan más de {mins} minutos en cargar. Antes era cuestión de "
         "segundos. La degradación de rendimiento está afectando a nuestros analistas."),
        ("Latencia alta en la API",
         "Las respuestas de {prod} superan los 8 segundos de latencia de forma constante. "
         "Necesitamos que esto se investigue porque rompe nuestras integraciones."),
    ],
    "Facturación": [
        ("Discrepancia en la factura del mes",
         "La factura de este periodo no coincide con lo acordado. Aparecen cargos que no "
         "reconocemos. Solicitamos revisión y, de proceder, una nota de crédito."),
        ("Solicitud de nota de crédito por incidencias",
         "Dado el tiempo de inactividad acumulado este trimestre, solicitamos formalmente una "
         "nota de crédito conforme al SLA de nuestro contrato."),
        ("Duda sobre el cobro de asientos adicionales",
         "Vemos un incremento en el monto facturado. ¿Pueden detallar el desglose de asientos "
         "y el prorrateo aplicado este mes?"),
    ],
    "Solicitud de función": [
        ("Necesitamos exportar reportes a Excel",
         "Nuestro equipo necesita exportar los tableros de {prod} a Excel y PDF de forma nativa. "
         "Hoy lo hacemos manualmente y consume mucho tiempo. ¿Está en el roadmap?"),
        ("Reportes personalizados por área",
         "Solicitamos poder crear reportes personalizados y programar su envío. Es una función "
         "clave para que nuestra dirección adopte la herramienta."),
        ("Tableros con métricas en tiempo real",
         "Queremos tableros con métricas en tiempo real. La actualización actual cada hora se "
         "queda corta para nuestro caso de uso."),
    ],
    "Integración / API": [
        ("Error de autenticación con el webhook",
         "Nuestro webhook hacia {prod} devuelve 401 de forma aleatoria. Revisamos las llaves y "
         "parecen correctas. ¿Pueden ayudarnos a depurar la autenticación?"),
        ("Documentación de la API incompleta",
         "Estamos integrando {prod} con nuestro ERP y faltan ejemplos para el endpoint de "
         "facturación. ¿Tienen una guía actualizada?"),
    ],
    "Consulta / Cómo hacer": [
        ("¿Cómo asigno roles a un nuevo usuario?",
         "Estamos sumando gente al equipo y queremos entender cómo asignar permisos por rol en "
         "{prod}. ¿Hay una guía paso a paso?"),
        ("Configuración de notificaciones",
         "Quisiéramos ajustar las notificaciones por correo de {prod}. ¿Dónde se configura el "
         "umbral de alertas?"),
    ],
    "Onboarding": [
        ("Sesión de capacitación inicial",
         "Como cuenta nueva, nos gustaría agendar la capacitación de onboarding para nuestro "
         "equipo en {prod}. ¿Cuál es la disponibilidad esta semana?"),
        ("Migración de datos inicial",
         "Necesitamos ayuda para importar nuestros datos históricos a {prod} durante el "
         "arranque. ¿Cuentan con una plantilla de carga?"),
    ],
    "Bug / Error": [
        ("Los filtros del tablero no se guardan",
         "Al aplicar filtros en los reportes de {prod} y recargar, se pierden. Parece un bug "
         "reciente; antes funcionaba bien."),
        ("Error al exportar datos",
         "Al intentar exportar desde {prod} aparece un mensaje de error genérico y el archivo "
         "sale vacío. Adjuntamos captura."),
    ],
}

# Distribución de categorías por arco narrativo (peso relativo).
CATEGORY_WEIGHTS = {
    "risk_loud": {  # ACME en Q2: outages, rendimiento, facturación (créditos)
        "Q1": {"Caída del servicio": 1, "Rendimiento": 1, "Consulta / Cómo hacer": 2,
               "Solicitud de función": 1, "Facturación": 1, "Bug / Error": 1},
        "Q2": {"Caída del servicio": 5, "Rendimiento": 4, "Facturación": 3,
               "Bug / Error": 2, "Consulta / Cómo hacer": 1},
    },
    "expansion": {  # GLOBEX: cuenta sana y creciendo -> funciones, cómo-hacer
        "Q1": {"Consulta / Cómo hacer": 2, "Solicitud de función": 2, "Integración / API": 1},
        "Q2": {"Solicitud de función": 3, "Integración / API": 2, "Consulta / Cómo hacer": 2},
    },
    "friction": {  # INITECH: sube fricción con reportes/dashboards
        "Q1": {"Solicitud de función": 2, "Bug / Error": 1, "Consulta / Cómo hacer": 2},
        "Q2": {"Solicitud de función": 4, "Bug / Error": 3, "Rendimiento": 1},
    },
    "new_logo": {  # UMBRELLA: onboarding e integración
        "Q1": {"Onboarding": 4, "Consulta / Cómo hacer": 3, "Integración / API": 2},
        "Q2": {"Consulta / Cómo hacer": 3, "Integración / API": 2, "Solicitud de función": 1},
    },
    "stable_star": {  # SOYLENT: poco volumen, cómo-hacer
        "Q1": {"Consulta / Cómo hacer": 3, "Solicitud de función": 1},
        "Q2": {"Consulta / Cómo hacer": 3, "Solicitud de función": 1},
    },
    "silent_churn": {  # HOOLI: casi sin tickets (desenganche)
        "Q1": {"Consulta / Cómo hacer": 2, "Bug / Error": 1},
        "Q2": {"Consulta / Cómo hacer": 1},
    },
}

# Volumen base de tickets por mes según arco (lista de Q1 y Q2)
TICKET_VOLUME = {
    "risk_loud":    {"Q1": (2, 4), "Q2": (9, 14)},
    "expansion":    {"Q1": (3, 5), "Q2": (4, 7)},
    "friction":     {"Q1": (3, 5), "Q2": (6, 9)},
    "new_logo":     {"Q1": (4, 7), "Q2": (3, 5)},
    "stable_star":  {"Q1": (1, 3), "Q2": (1, 3)},
    "silent_churn": {"Q1": (1, 2), "Q2": (0, 1)},
}

CHANNELS = ["Correo", "Portal", "Chat", "Teléfono"]
PRIORITY_BY_CAT = {
    "Caída del servicio": ["P1", "P1", "P2"],
    "Rendimiento": ["P2", "P2", "P3"],
    "Facturación": ["P3", "P2"],
    "Solicitud de función": ["P4", "P3"],
    "Integración / API": ["P3", "P2"],
    "Consulta / Cómo hacer": ["P4", "P3"],
    "Onboarding": ["P3", "P4"],
    "Bug / Error": ["P2", "P3"],
}


def weighted_choice(weights: dict):
    cats, ws = zip(*weights.items())
    return random.choices(cats, weights=ws, k=1)[0]


def csat_for(story, quarter):
    """CSAT del ticket (1-5) coherente con la salud de la cuenta."""
    if story == "risk_loud" and quarter == "Q2-2026":
        return random.choices([1, 2, 3], weights=[4, 4, 2])[0]
    if story in ("stable_star", "expansion"):
        return random.choices([4, 5], weights=[3, 5])[0]
    if story == "friction" and quarter == "Q2-2026":
        return random.choices([2, 3, 4], weights=[3, 4, 2])[0]
    return random.choices([3, 4, 5], weights=[3, 4, 3])[0]


# ---------------------------------------------------------------------------
# 3) GENERACIÓN
# ---------------------------------------------------------------------------
def month_dates(m):
    y, mo = map(int, m.split("-"))
    start = date(y, mo, 1)
    nxt = date(y + (mo == 12), (mo % 12) + 1, 1)
    return start, nxt - timedelta(days=1)


def build():
    db_path = os.path.join(OUT, "qbr.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript(
        """
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY, name TEXT, industry TEXT, country TEXT,
            region TEXT, segment TEXT, csm_owner TEXT, contract_start TEXT,
            renewal_quarter TEXT, seats_contracted INTEGER
        );
        CREATE TABLE subscriptions (
            account_id TEXT, product TEXT, plan_tier TEXT, seats INTEGER,
            monthly_price REAL
        );
        CREATE TABLE invoices (
            invoice_id TEXT PRIMARY KEY, account_id TEXT, invoice_date TEXT,
            quarter TEXT, amount REAL, status TEXT
        );
        CREATE TABLE usage_monthly (
            account_id TEXT, month TEXT, active_users INTEGER, api_calls INTEGER,
            logins INTEGER, feature_adoption_score REAL
        );
        CREATE TABLE support_tickets (
            ticket_id TEXT PRIMARY KEY, account_id TEXT, created_date TEXT,
            quarter TEXT, channel TEXT, category TEXT, priority TEXT,
            status TEXT, csat INTEGER, subject TEXT, description TEXT
        );
        CREATE TABLE account_health (
            account_id TEXT, quarter TEXT, nps INTEGER, csat_avg REAL,
            health_score INTEGER
        );
        """
    )

    accounts_rows, subs_rows, inv_rows, usage_rows, ticket_rows, health_rows = (
        [], [], [], [], [], [])
    all_tickets_for_docs = {a["account_id"]: [] for a in ACCOUNTS}
    tnum = 1
    inum = 1

    for a in ACCOUNTS:
        accounts_rows.append((a["account_id"], a["name"], a["industry"], a["country"],
                              a["region"], a["segment"], a["csm_owner"],
                              a["contract_start"], a["renewal_quarter"], a["seats"]))

        # suscripciones: repartir el MRR entre productos
        splits = [0.55, 0.30, 0.15]
        for prod, sp in zip(PRODUCTS, splits):
            tier = "Enterprise" if a["segment"] == "Enterprise" else "Pro"
            subs_rows.append((a["account_id"], prod, tier,
                              int(a["seats"] * sp), round(a["mrr_base"] * sp, 2)))

        # salud trimestral
        for q, (nps, csat, hs) in HEALTH[a["account_id"]].items():
            health_rows.append((a["account_id"], q, nps, csat, hs))

        for i, m in enumerate(MONTHS):
            q = quarter_of(m)
            qkey = "Q1" if m in Q1 else "Q2"
            mstart, mend = month_dates(m)

            # ---- facturación mensual ----
            amount = round(a["mrr_base"] * a["rev_arc"][i], 2)
            # ACME en Q2: parte como nota de crédito (status credited)
            if a["story"] == "risk_loud" and qkey == "Q2":
                status = "credited" if random.random() < 0.5 else "paid"
            elif a["story"] == "silent_churn":
                status = "paid"
            else:
                status = random.choices(["paid", "overdue"], weights=[9, 1])[0]
            inv_rows.append((f"INV-{inum:05d}", a["account_id"],
                             mstart.isoformat(), q, amount, status))
            inum += 1

            # ---- uso mensual ----
            base_users = a["seats"]
            active = int(base_users * a["usage_arc"][i] * random.uniform(0.85, 0.98))
            logins = int(active * random.uniform(8, 22))
            api_calls = int(active * random.uniform(200, 1200) *
                            (1.4 if "API" in PRODUCTS[2] and a["story"] == "expansion" else 1.0))
            adoption = round(min(1.0, a["usage_arc"][i] * random.uniform(0.6, 0.95)), 2)
            usage_rows.append((a["account_id"], m, active, api_calls, logins, adoption))

            # ---- tickets ----
            lo, hi = TICKET_VOLUME[a["story"]][qkey]
            n_tickets = random.randint(lo, hi)
            weights = CATEGORY_WEIGHTS[a["story"]][qkey]
            for _ in range(n_tickets):
                cat = weighted_choice(weights)
                subj, body = random.choice(TICKET_TEMPLATES[cat])
                prod = random.choice(PRODUCTS)
                body = body.format(prod=prod, mins=random.choice([20, 45, 90, 120]),
                                   nth=random.choice(["segunda", "tercera", "cuarta"]))
                prio = random.choice(PRIORITY_BY_CAT[cat])
                cday = fake.date_between(start_date=mstart, end_date=mend)
                status_t = random.choices(["Resuelto", "En curso", "Escalado"],
                                          weights=[6, 2, 2])[0]
                csat = csat_for(a["story"], q)
                tid = f"TCK-{tnum:05d}"
                ticket_rows.append((tid, a["account_id"], cday.isoformat(), q,
                                    random.choice(CHANNELS), cat, prio, status_t,
                                    csat, subj, body))
                all_tickets_for_docs[a["account_id"]].append(
                    (tid, cday.isoformat(), cat, prio, subj, body))
                tnum += 1

    cur.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?,?)", accounts_rows)
    cur.executemany("INSERT INTO subscriptions VALUES (?,?,?,?,?)", subs_rows)
    cur.executemany("INSERT INTO invoices VALUES (?,?,?,?,?,?)", inv_rows)
    cur.executemany("INSERT INTO usage_monthly VALUES (?,?,?,?,?,?)", usage_rows)
    cur.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?,?,?,?,?)", ticket_rows)
    cur.executemany("INSERT INTO account_health VALUES (?,?,?,?,?)", health_rows)
    con.commit()

    # ---- exportar CSVs ----
    def dump(table, cols):
        rows = cur.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()
        with open(os.path.join(OUT, f"{table}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)

    dump("accounts", ["account_id","name","industry","country","region","segment",
                      "csm_owner","contract_start","renewal_quarter","seats_contracted"])
    dump("subscriptions", ["account_id","product","plan_tier","seats","monthly_price"])
    dump("invoices", ["invoice_id","account_id","invoice_date","quarter","amount","status"])
    dump("usage_monthly", ["account_id","month","active_users","api_calls","logins","feature_adoption_score"])
    dump("support_tickets", ["ticket_id","account_id","created_date","quarter","channel",
                             "category","priority","status","csat","subject","description"])
    dump("account_health", ["account_id","quarter","nps","csat_avg","health_score"])

    con.close()
    return all_tickets_for_docs


# ---------------------------------------------------------------------------
# 4) NOTAS DE CUENTA (texto cualitativo en español, para el RAG)
# ---------------------------------------------------------------------------
NOTES = {
    "ACME": """# Nota de cuenta — Acme Corporation

**CSM:** Valeria Núñez · **Segmento:** Enterprise · **Renovación:** Q3-2026

## Contexto del trimestre (Q2-2026)
Trimestre crítico. Tras una serie de **caídas del servicio en producción** durante
abril y mayo, la relación se ha deteriorado de forma notable. El equipo de
operaciones del cliente reportó múltiples incidentes P1 y exigió reportes
post-incidente formales. Se emitieron **notas de crédito** por incumplimiento de SLA,
lo que explica la caída en los ingresos reconocidos del trimestre.

## Cambios clave
- **Salida del champion:** nuestro principal patrocinador interno, el Director de
  TI, dejó la empresa en mayo. El nuevo responsable es escéptico y favorece
  consolidar proveedores.
- **Presión competitiva:** Acme está **evaluando activamente a un competidor**
  ("StratusOne") que les ofreció una migración asistida y descuento agresivo.

## Riesgos
- Riesgo de **churn ALTO** de cara a la renovación de Q3-2026.
- La caída de uso (usuarios activos a la baja) sugiere desenganche real, no solo enojo.

## Próximos pasos recomendados
- Escalar a un *executive sponsor* de nuestro lado y agendar un *executive business
  review* presencial.
- Plan de remediación técnica con compromisos de disponibilidad medibles.
""",
    "GLOBEX": """# Nota de cuenta — Globex S.A.

**CSM:** Diego Hernández · **Segmento:** Enterprise · **Renovación:** Q4-2026

## Contexto del trimestre (Q2-2026)
Cuenta en **expansión**. La adopción de Nimbus Analytics creció de forma sostenida
y el equipo del cliente sumó asientos en mayo. La mayoría de los tickets son
**solicitudes de función** e integraciones, señal típica de un usuario comprometido
que quiere exprimir más valor del producto.

## Oportunidades
- **Upsell** del módulo de API (Nimbus Connect): ya están integrando con su ERP.
- Candidata fuerte a **caso de éxito / referencia** para retail en LATAM.

## Riesgos
- Bajos. Vigilar que el roadmap de reportes personalizados avance para no frenar el
  entusiasmo.
""",
    "INITECH": """# Nota de cuenta — Initech

**CSM:** Diego Hernández · **Segmento:** Mid-Market · **Renovación:** Q1-2027

## Contexto del trimestre (Q2-2026)
Ingresos planos pero **fricción creciente** alrededor del módulo de **reportes y
tableros**: aumentaron los tickets de bugs (filtros que no se guardan, exportaciones
vacías) y las solicitudes de reportes personalizados. La CSAT bajó ligeramente.

## Riesgos
- Riesgo **MEDIO**: la insatisfacción está concentrada en una sola área de producto.
  Si se atiende el roadmap de reportes, la cuenta se estabiliza.

## Próximos pasos
- Conectar al cliente con el PM de Analytics para una sesión de feedback.
- Priorizar la corrección de los bugs de exportación reportados.
""",
    "UMBRELLA": """# Nota de cuenta — Umbrella Corp

**CSM:** Valeria Núñez · **Segmento:** Mid-Market · **Renovación:** Q1-2027

## Contexto del trimestre (Q2-2026)
**Logo nuevo** (arranque en enero). El onboarding avanzó bien: tras un Q1 de
capacitación y migración de datos, en Q2 el uso ya rampó por encima de lo
contratado y la salud mejoró. Tickets ahora son mayormente de configuración e
integración, propios de una cuenta que está adoptando.

## Oportunidades
- Buen momento para asegurar un *quick win* documentado antes de la renovación.

## Riesgos
- Bajos, pero es cuenta joven: mantener cadencia de acompañamiento.
""",
    "SOYLENT": """# Nota de cuenta — Soylent Industries

**CSM:** Diego Hernández · **Segmento:** Enterprise · **Renovación:** Q2-2027

## Contexto del trimestre (Q2-2026)
**Cuenta estrella y estable.** Ingresos y uso constantes en niveles altos, CSAT y NPS
sobresalientes, muy pocos tickets (y los que hay son consultas de configuración). Es
nuestra mejor **cuenta de referencia** en el sector de alimentos y bebidas.

## Oportunidades
- Solicitar testimonio / *case study* y participación en eventos.

## Riesgos
- Mínimos. Evitar la complacencia: mantener revisiones trimestrales.
""",
    "HOOLI": """# Nota de cuenta — Hooli

**CSM:** Valeria Núñez · **Segmento:** Enterprise · **Renovación:** Q3-2026

## Contexto del trimestre (Q2-2026)
**Atención: churn silencioso.** Los ingresos se mantienen planos y la cuenta *parece*
sana en facturación, pero el **uso se ha desplomado** trimestre con trimestre
(usuarios activos, logins y adopción a la baja). Casi no abren tickets, lo que en este
caso **no** es buena señal: indica desenganche, no satisfacción.

## Cambios clave
- No hay un *executive sponsor* claro desde hace meses.
- El equipo que originalmente impulsó la herramienta se reorganizó.

## Riesgos
- Riesgo de **churn ALTO y poco visible** de cara a Q3-2026. Es el tipo de cuenta que
  se pierde "sin avisar" porque los números de facturación no disparan alarmas.

## Próximos pasos
- Campaña de re-enganche y re-onboarding.
- Identificar y cultivar un nuevo champion con urgencia.
""",
}


def write_docs(tickets_by_acct):
    """Escribe notas de cuenta y un corpus combinado (notas + tickets) para el RAG."""
    name_by_id = {a["account_id"]: a["name"] for a in ACCOUNTS}
    for aid, note in NOTES.items():
        with open(os.path.join(NOTES_DIR, f"{aid.lower()}_nota.md"), "w", encoding="utf-8") as f:
            f.write(note)

    # corpus combinado: un .md por cuenta con su nota + todos sus tickets
    for aid, tickets in tickets_by_acct.items():
        lines = [NOTES[aid], "\n---\n", f"## Tickets de soporte — {name_by_id[aid]}\n"]
        for tid, cday, cat, prio, subj, body in sorted(tickets, key=lambda x: x[1]):
            lines.append(f"\n### [{tid}] {subj}\n"
                         f"*Fecha:* {cday} · *Categoría:* {cat} · *Prioridad:* {prio}\n\n"
                         f"{body}\n")
        with open(os.path.join(DOCS_DIR, f"{aid.lower()}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    tickets_by_acct = build()
    write_docs(tickets_by_acct)
    print("OK — datos generados en", OUT)
