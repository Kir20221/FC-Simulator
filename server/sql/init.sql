-- init.sql
-- ============================================================
-- Projet Drake — Schéma initial
-- ============================================================

-- ---------- Zone scénario --------------------------------------------------

CREATE TABLE scenario (
    id              BIGSERIAL PRIMARY KEY,
    nom             TEXT        NOT NULL,
    date_creation   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    statut_entites  TEXT        NOT NULL DEFAULT 'en_attente'
        CHECK (statut_entites IN ('en_attente', 'en_cours', 'termine', 'echec')),

    r_star  DOUBLE PRECISION NOT NULL,
    fp      DOUBLE PRECISION NOT NULL,
    ne      DOUBLE PRECISION NOT NULL,
    fl      DOUBLE PRECISION NOT NULL,
    fi      DOUBLE PRECISION NOT NULL,
    fc      DOUBLE PRECISION NOT NULL,
    l_drake DOUBLE PRECISION NOT NULL,

    nombre_systemes BIGINT NOT NULL CHECK (nombre_systemes > 0)
);

CREATE TABLE simulation (
    id              BIGSERIAL PRIMARY KEY,
    scenario_id     BIGINT      NOT NULL REFERENCES scenario(id),
    date_lancement  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    statut          TEXT        NOT NULL DEFAULT 'en_attente'
        CHECK (statut IN ('en_attente', 'en_cours', 'terminee', 'echec')),
    progression     DOUBLE PRECISION NOT NULL DEFAULT 0.0
        CHECK (progression BETWEEN 0.0 AND 1.0),
    seed            BIGINT      NOT NULL
);

CREATE INDEX idx_simulation_scenario ON simulation(scenario_id);

-- ---------- Entités ---------------------------------------------------------

CREATE TABLE galaxie (
    id          BIGSERIAL PRIMARY KEY,
    scenario_id BIGINT NOT NULL UNIQUE REFERENCES scenario(id)
);

CREATE TABLE systeme_solaire (
    id         BIGSERIAL PRIMARY KEY,
    galaxie_id BIGINT NOT NULL REFERENCES galaxie(id),
    position_x DOUBLE PRECISION NOT NULL,
    position_y DOUBLE PRECISION NOT NULL,
    position_z DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_systeme_galaxie ON systeme_solaire(galaxie_id);

CREATE TABLE etoile (
    id         BIGSERIAL PRIMARY KEY,
    systeme_id BIGINT NOT NULL REFERENCES systeme_solaire(id)
);

CREATE INDEX idx_etoile_systeme ON etoile(systeme_id);

CREATE TABLE planete (
    id         BIGSERIAL PRIMARY KEY,
    systeme_id BIGINT NOT NULL REFERENCES systeme_solaire(id),
    etoile_id  BIGINT NOT NULL REFERENCES etoile(id)
);

CREATE INDEX idx_planete_systeme ON planete(systeme_id);
CREATE INDEX idx_planete_etoile  ON planete(etoile_id);

-- ---------- Événements ------------------------------------------------------

CREATE TABLE type_evenement (
    id      SMALLSERIAL PRIMARY KEY,
    libelle TEXT NOT NULL UNIQUE
);

INSERT INTO type_evenement (libelle) VALUES
    ('emergence_vie'),
    ('emergence_civilisation'),
    ('extinction'),
    ('expansion'),
    ('contact');

CREATE TABLE evenement (
    id                 BIGSERIAL PRIMARY KEY,
    simulation_id      BIGINT  NOT NULL REFERENCES simulation(id),
    timecode           BIGINT  NOT NULL,
    type_evenement_id  SMALLINT NOT NULL REFERENCES type_evenement(id),
    entite_id          BIGINT  NOT NULL,
    entite_type        TEXT    NOT NULL
        CHECK (entite_type IN ('galaxie', 'systeme', 'etoile', 'planete')),
    payload            JSONB
);

CREATE INDEX idx_evenement_sim_time ON evenement(simulation_id, timecode);
CREATE INDEX idx_evenement_entite   ON evenement(entite_type, entite_id);


-- Table dataset : métadonnées des datasets d'entraînement.
-- Les fichiers parquet eux-mêmes vivent sur disque (volume monté),
-- pas en DB.
CREATE TABLE IF NOT EXISTS dataset (
    nom            TEXT PRIMARY KEY,
    date_creation  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    nb_systemes    INTEGER NOT NULL,
    seed           INTEGER NOT NULL,
    chemin_planets TEXT,
    chemin_events  TEXT,
    taille_octets  BIGINT NOT NULL,
    -- Stats calculées au moment de la génération (snapshot, pas recalculé).
    stats          JSONB NOT NULL
);
 
-- Index sur la date pour lister du plus récent au plus ancien.
CREATE INDEX IF NOT EXISTS idx_dataset_date_creation
    ON dataset (date_creation DESC);
 