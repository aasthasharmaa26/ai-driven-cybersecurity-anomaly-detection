from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import Boolean

from database.db import Base

class Alert(Base):

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)

    log = Column(String)

    cleaned_log = Column(String)

    reconstruction_error = Column(Float)

    anomaly_detected = Column(Boolean)

    threat_type = Column(String)

    severity = Column(String)

    risk_score = Column(Integer)

    mitre_attack = Column(String)