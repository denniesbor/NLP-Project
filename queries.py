import os
import sqlalchemy as db
from dotenv import load_dotenv
from urllib.parse import quote
#load secret files
load_dotenv()

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = 3306
DATABASE = "nlp"
TABLENAME = "us_house_reps"

engine = db.create_engine(f"mysql+pymysql://{USER}:%s@{HOST}:{PORT}/{DATABASE}" % quote(PASSWORD))
connection = engine.connect()
metadata = db.MetaData()
housereps = db.Table(f'{TABLENAME}', metadata, autoload=True, autoload_with=engine)

print(housereps.columns.keys())

query = db.select([housereps])
ResultProxy = connection.execute(query)
ResultSet = ResultProxy.fetchall()
print(ResultSet[:10])