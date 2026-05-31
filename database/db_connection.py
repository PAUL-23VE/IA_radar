"""
database/db_connection.py
Conexión a PostgreSQL y consultas para buscar vehículos por placa.
"""

import psycopg2
import psycopg2.extras
from datetime import datetime

# ----------------------------------------------------------------
#  CONFIGURACIÓN — ajusta estos valores a tu entorno local
# ----------------------------------------------------------------
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "placas_db",
    "user":     "postgres",
    "password": "12345678"          # Cambia por tu contraseña
}


def obtener_conexion():
    """Devuelve una conexión activa a PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)


# ----------------------------------------------------------------
#  CONSULTA PRINCIPAL: buscar vehículo por placa
# ----------------------------------------------------------------
def buscar_vehiculo(placa: str) -> dict | None:
    """
    Recibe el string de la placa (ej: 'ABC-1234')
    y devuelve un diccionario con todos los datos del vehículo
    y su propietario, o None si no existe.
    """
    placa = placa.upper().strip()

    sql = """
        SELECT
            v.placa,
            v.marca,
            v.modelo,
            v.anio,
            v.color,
            v.tipo,
            v.cilindraje,
            p.nombres || ' ' || p.apellidos  AS propietario,
            p.cedula,
            p.email,
            p.telefono,
            p.direccion
        FROM vehiculos v
        LEFT JOIN propietarios p ON v.propietario_id = p.id
        WHERE v.placa = %s
    """

    try:
        with obtener_conexion() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(sql, (placa,))
                fila = cur.fetchone()
                return dict(fila) if fila else None

    except psycopg2.Error as e:
        print(f"[DB ERROR] {e}")
        return None


# ----------------------------------------------------------------
#  REGISTRAR MULTA
# ----------------------------------------------------------------
def registrar_multa(placa: str, velocidad: float,
                    clasificacion: str, dias: int) -> bool:
    """
    Guarda una multa en la tabla 'multas'.
    Retorna True si se guardó correctamente.
    """
    sql = """
        INSERT INTO multas (vehiculo_id, placa, velocidad_kmh,
                            clasificacion, dias_sin_ingreso)
        SELECT v.id, %s, %s, %s, %s
        FROM vehiculos v WHERE v.placa = %s
    """

    try:
        with obtener_conexion() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (placa, velocidad, clasificacion, dias, placa))
            conn.commit()
        return True

    except psycopg2.Error as e:
        print(f"[DB ERROR al registrar multa] {e}")
        return False


# ----------------------------------------------------------------
#  MOSTRAR DATOS EN CONSOLA (util para debug)
# ----------------------------------------------------------------
def imprimir_vehiculo(datos: dict):
    """Imprime los datos del vehículo de forma legible."""
    if not datos:
        print("❌  Vehículo NO encontrado en la base de datos.")
        return

    print("\n" + "="*50)
    print("  VEHÍCULO ENCONTRADO")
    print("="*50)
    print(f"  Placa       : {datos['placa']}")
    print(f"  Marca       : {datos['marca']}")
    print(f"  Modelo      : {datos['modelo']}")
    print(f"  Año         : {datos['anio']}")
    print(f"  Color       : {datos['color']}")
    print(f"  Tipo        : {datos['tipo']}")
    print(f"  Cilindraje  : {datos['cilindraje']}")
    print("-"*50)
    print(f"  Propietario : {datos['propietario']}")
    print(f"  Cédula      : {datos['cedula']}")
    print(f"  Email       : {datos['email']}")
    print(f"  Teléfono    : {datos['telefono']}")
    print(f"  Dirección   : {datos['direccion']}")
    print("="*50 + "\n")


# ----------------------------------------------------------------
#  TEST RÁPIDO (ejecutar directamente: python database/db_connection.py)
# ----------------------------------------------------------------
if __name__ == "__main__":
    placas_prueba = ["ABC-1234", "XYZ-4567", "AAA-0000"]
    for p in placas_prueba:
        print(f"\nBuscando placa: {p}")
        resultado = buscar_vehiculo(p)
        imprimir_vehiculo(resultado)
