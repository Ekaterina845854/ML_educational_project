import os
import hashlib
import logging
from fastapi import FastAPI, Depends, HTTPException
from typing import List
from sqlalchemy import Column, Integer, String, create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from catboost import CatBoostClassifier
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Определение базовой модели для SQLAlchemy
Base = declarative_base()

# Определение модели Post
class Post(Base):
    __tablename__ = "posts"
    post_id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    topic = Column(String)

# Определение модели для ответа
class PostGet(BaseModel):
    post_id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

# Настройка подключения к базе данных
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")  
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

SALT = "s"
CONTROL_GROUP = "control"
TEST_GROUP = "test"

@lru_cache(maxsize=2)
def get_model(model_name: str) -> CatBoostClassifier:
    model_path = f"./models/{model_name}"
    model = CatBoostClassifier()
    try:
        model.load_model(model_path, format="cbm")
        logger.info(f"Model loaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Model {model_name} could not be loaded.")
    return model

# Функция для определения группы
def get_exp_group(user_id: int) -> str:
    hash_input = f"{user_id}{SALT}".encode()
    user_hash = hashlib.md5(hash_input).hexdigest()
    hash_value = int(user_hash, 16)
    group = CONTROL_GROUP if hash_value % 2 == 0 else TEST_GROUP
    logger.info(f"User {user_id} assigned to group: {group}")
    return group

# Проверка доступности базы данных
def is_database_available() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

# Проверка наличия таблицы
def is_table_exists(table_name: str) -> bool:
    inspector = inspect(engine)
    return inspector.has_table(table_name)

# Функция для получения сессии базы данных
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    except OperationalError as e:
        logger.error(f"Database session failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")
    finally:
        db.close()

def load_data_from_source(table_name: str) -> pd.DataFrame:
    if is_database_available():
        query_public = f'SELECT * FROM "public"."{table_name}"'
        try:
            with engine.connect().execution_options(stream_results=True) as conn:
                chunks_public = pd.read_sql(query_public, conn, chunksize=10000)
                data_public = pd.concat(chunks_public, ignore_index=True)
                if not data_public.empty:
                    return data_public
                else:
                    print("DataFrame is empty.")
        except Exception as e:
            logger.error(f"Failed to load data from public schema: {e}")

    # Если данные не найдены в базе данных, проверяем CSV файл
    csv_path = os.path.join("./data/", f"{table_name}.csv")
    if os.path.exists(csv_path):
        logger.info(f"Loading data from CSV file: {csv_path}")
        return pd.read_csv(csv_path)
    
    logger.warning(f"File not found: {csv_path}")
    return pd.DataFrame()



# Загрузка пользовательских признаков
def load_user_features(user_id: int, table_name: str) -> pd.DataFrame:
    data = load_data_from_source(table_name)
    logger.info(f"Loaded data columns for {table_name}: {data.columns.tolist()}")

    if 'user_id' not in data.columns:
        logger.error(f"'user_id' column not found in data: {data.columns.tolist()}")
        raise ValueError(f"'user_id' column not found in data: {data.columns.tolist()}")

    user_data = data[data["user_id"] == user_id]
    logger.info(f"User {user_id} features: {user_data}")

    return user_data

# Функция для получения рекомендаций
def get_recommendations(user_features: pd.DataFrame, model: CatBoostClassifier, limit: int, db: Session) -> List[PostGet]:
    if user_features.empty:
        logger.warning("User features are empty. Cannot generate recommendations.")
        raise HTTPException(status_code=404, detail="No user features available.")

    predictions = model.predict_proba(user_features)[:, 1]
    logger.info(f"Predictions: {predictions}")

    if not predictions.any():
        logger.warning("Model returned no positive predictions.")
        raise HTTPException(status_code=404, detail="No recommendations available.")

    top_post_indices = predictions.argsort()[-limit:][::-1]
    post_ids = user_features.iloc[top_post_indices]["post_id"].tolist()
    logger.info(f"Predicted post IDs: {post_ids}")

    post_map = {}
    if db and is_table_exists("posts"):
        posts = db.query(Post).filter(Post.post_id.in_(post_ids)).all()
        logger.info(f"Found posts in DB: {[post.post_id for post in posts]}")
        post_map = {post.post_id: post for post in posts}
    else:
        posts_df = load_data_from_source("posts")
        if not posts_df.empty:
            post_map = {row.post_id: row for row in posts_df.itertuples() if row.post_id in post_ids}

    if not post_map:
        logger.warning("No posts found matching the recommendations.")
        raise HTTPException(status_code=404, detail="No matching posts found.")

    recommendations = [
        PostGet(post_id=post_id, text=post_map[post_id].text, topic=post_map[post_id].topic)
        for post_id in post_ids if post_id in post_map
    ]
    logger.info(f"Generated {len(recommendations)} recommendations.")
    return recommendations


# Эндпоинт для получения рекомендаций
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, limit: int = 5, db: Session = Depends(get_db)) -> Response:
    logger.info(f"Request received for user ID: {id} with limit: {limit}")
    
    if db is None and not is_database_available():
        logger.error("Database connection failed.")
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    exp_group = get_exp_group(id)
    table_name = "data_control" if exp_group == CONTROL_GROUP else "data_test"
    user_features = load_user_features(id, table_name)

    if user_features.empty:
        logger.warning("User features not found.")
        raise HTTPException(status_code=404, detail="User features not found.")

    model = get_model("model_control" if exp_group == CONTROL_GROUP else "model_test")
    recommendations = get_recommendations(user_features, model, limit, db)

    return Response(exp_group=exp_group, recommendations=recommendations)


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)

