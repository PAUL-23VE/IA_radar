-- ============================================================
--  BASE DE DATOS: placas_db
--  Ejecutar: psql -U postgres -f database/schema.sql
-- ============================================================

-- Crear la base de datos (ejecutar como superusuario)
CREATE DATABASE placas_db
    WITH ENCODING 'UTF8'
    LC_COLLATE 'es_EC.UTF-8'
    LC_CTYPE 'es_EC.UTF-8'
    TEMPLATE template0;

\connect placas_db;

-- ============================================================
--  TABLA: propietarios
-- ============================================================
CREATE TABLE propietarios (
    id          SERIAL PRIMARY KEY,
    cedula      VARCHAR(10)  NOT NULL UNIQUE,
    nombres     VARCHAR(100) NOT NULL,
    apellidos   VARCHAR(100) NOT NULL,
    email       VARCHAR(150) NOT NULL,
    telefono    VARCHAR(15),
    direccion   TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- ============================================================
--  TABLA: vehiculos
-- ============================================================
CREATE TABLE vehiculos (
    id              SERIAL PRIMARY KEY,
    placa           VARCHAR(8)   NOT NULL UNIQUE,   -- Formato Ecuador: ABC-1234
    marca           VARCHAR(50)  NOT NULL,
    modelo          VARCHAR(80)  NOT NULL,
    anio            INTEGER      NOT NULL,
    color           VARCHAR(40)  NOT NULL,
    tipo            VARCHAR(30)  NOT NULL,           -- Sedan, SUV, Camioneta, etc.
    cilindraje      VARCHAR(10),
    propietario_id  INTEGER REFERENCES propietarios(id) ON DELETE SET NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
--  TABLA: multas  (se llena en tiempo de ejecución)
-- ============================================================
CREATE TABLE multas (
    id              SERIAL PRIMARY KEY,
    vehiculo_id     INTEGER REFERENCES vehiculos(id) ON DELETE CASCADE,
    placa           VARCHAR(8)  NOT NULL,
    velocidad_kmh   DECIMAL(5,2),
    clasificacion   VARCHAR(20),   -- 'normal', 'multa'
    dias_sin_ingreso INTEGER DEFAULT 0,
    timestamp       TIMESTAMP DEFAULT NOW()
);

-- ============================================================
--  ÍNDICES para búsqueda rápida por placa
-- ============================================================
CREATE INDEX idx_placa ON vehiculos(placa);
CREATE INDEX idx_multas_placa ON multas(placa);
CREATE INDEX idx_multas_timestamp ON multas(timestamp);
