import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App General
    DISTANCIA_REFERENCIA_METROS: float = 5.0
    
    # Camara / DroidCam
    IP_IPHONE: str = "192.168.9.129"
    PUERTO: str = "4747"
    
    # Correo / SMTP
    SMTP_HOST: str = "smtp.office365.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "dmanjarres6065@uta.edu.ec"
    SMTP_PASSWORD: str = "hqgqxvpwpzwjyxyl"
    SMTP_FROM: str = "dmanjarres6065@uta.edu.ec"
    ENVIO_INFRACCIONES_A: str = "davidmanjarres2004@gmail.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
