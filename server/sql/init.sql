-- init.sql
-- ============================================================
-- Projet FC-Simulator — Schéma initial
-- ============================================================

-- ---------- Zone scénario --------------------------------------------------

CREATE TABLE scenario (
    id              BIGSERIAL PRIMARY KEY,
    nom             TEXT        NOT NULL,
    date_creation   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    statut_entites  TEXT        NOT NULL DEFAULT 'en_attente'
        CHECK (statut_entites IN ('en_attente', 'en_cours', 'termine', 'echec')),

    masse_stellaire_moyenne     DOUBLE PRECISION NOT NULL CHECK (masse_stellaire_moyenne > 0),
    indice_tellurique           DOUBLE PRECISION NOT NULL CHECK (indice_tellurique BETWEEN 0 AND 1),
    planetes_par_systeme_moyen  DOUBLE PRECISION NOT NULL CHECK (planetes_par_systeme_moyen > 0),
    duree_simulation_Ga         DOUBLE PRECISION NOT NULL CHECK (duree_simulation_Ga > 0),

    nb_systemes BIGINT NOT NULL CHECK (nb_systemes > 0)
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
    id                      BIGSERIAL PRIMARY KEY,
    systeme_id              BIGINT NOT NULL REFERENCES systeme_solaire(id),
    star_type               TEXT             NOT NULL,
    star_temp_K             DOUBLE PRECISION NOT NULL,
    star_mass_solar         DOUBLE PRECISION NOT NULL,
    star_luminosity_solar   DOUBLE PRECISION NOT NULL,
    star_lifetime_Ga        DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_etoile_systeme ON etoile(systeme_id);

CREATE TABLE planete (
    id                      BIGSERIAL PRIMARY KEY,
    systeme_id              BIGINT NOT NULL REFERENCES systeme_solaire(id),
    etoile_id               BIGINT NOT NULL REFERENCES etoile(id),
    planet_distance_UA      DOUBLE PRECISION NOT NULL,
    planet_mass_terre       DOUBLE PRECISION NOT NULL,
    planet_radius_terre     DOUBLE PRECISION NOT NULL,
    planet_composition      TEXT             NOT NULL
);

CREATE INDEX idx_planete_systeme ON planete(systeme_id);
CREATE INDEX idx_planete_etoile  ON planete(etoile_id);

-- ---------- Événements ------------------------------------------------------

CREATE TABLE type_evenement (
    id      SMALLSERIAL PRIMARY KEY,
    libelle TEXT NOT NULL UNIQUE
);

INSERT INTO type_evenement (libelle) VALUES
    ('life_apparition'),
    ('star_main_sequence_end');

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

-- table 'model' pour les métadonnées des modèles entraînés.
CREATE TABLE IF NOT EXISTS model (
    nom               TEXT PRIMARY KEY,
    date_creation     TIMESTAMPTZ NOT NULL,
    dataset_nom       TEXT NOT NULL,                  -- nom du dataset utilisé pour l'entraînement
    chemin_fichier    TEXT NOT NULL,                  -- chemin du .pth dans le volume
    taille_octets     BIGINT NOT NULL,
    duree_entrainement_s  DOUBLE PRECISION NOT NULL,
    -- Hyperparamètres et métriques sérialisés en JSONB.
    -- Permet d'ajouter des champs au schéma sans migration.
    training_config   JSONB NOT NULL,                 -- TrainingConfig.__dict__
    tpp_config        JSONB NOT NULL,                 -- TPPConfig.__dict__
    metrics_par_epoch JSONB NOT NULL,                 -- list[dict]
    final_train_loss  DOUBLE PRECISION NOT NULL,
    final_val_loss    DOUBLE PRECISION NOT NULL
);
