import os
from fastapi import FastAPI, HTTPException, Depends, Request, status, Path, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import httpx
import datetime
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import requests

# Load environment variables
load_dotenv()

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    yield
    # Shutdown tasks
    scheduler.shutdown()
    logger.info("Application shutting down")

app = FastAPI(lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google OAuth setup
CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": ["http://localhost:5001/callback"],
        "javascript_origins": ["http://localhost:5001"]
    }
}

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"openid": "OpenID Connect", "email": "Email", "profile": "Profile"}
)

# Database models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    google_id = Column(String, unique=True, index=True)

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(String, primary_key=True, index=True)
    video_id = Column(String, unique=True, index=True)
    title = Column(String)
    summary = Column(Text)
    user_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database initialization
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized")
        
        # Add test data only if the summaries table is empty
        db = SessionLocal()
        if db.query(Summary).count() == 0:
            add_test_data(db)
        db.close()
        
        # Check existing summaries
        db = SessionLocal()
        summaries = db.query(Summary).all()
        logger.info(f"Total {len(summaries)} summaries found")
        for summary in summaries:
            logger.info(f"Summary ID: {summary.id}, Title: {summary.title}")
        db.close()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class VideoURL(BaseModel):
    url: str

class Question(BaseModel):
    summary_id: str
    question: str

class DeleteSummaryRequest(BaseModel):
    summary_id: str

# Helper functions
def get_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        return None

def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an academic assistant that summarizes YouTube video transcripts. Provide a structured summary with the following format:\n\nTitle: [Title here]\n\nAbstract: [Abstract here]\n\nIntroduction: [Introduction here]\n\nKey Points:\n- [Point 1]\n- [Point 2]\n- [Point 3]\n\nConclusion: [Conclusion here]\n\nUse formal academic language and structure."},
                {"role": "user", "content": f"Please summarize the following YouTube video transcript in an academic style:\n\n{text}"}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        return None

# Test data addition function
def add_test_data(db: Session):
    test_summaries = [
        Summary(id="unique-id-1", video_id="test1", title="Missing Field Error", summary="This is a test summary 1"),
        Summary(id="unique-id-2", video_id="test2", title="VS Code'dan GitHub'a", summary="This is a test summary 2"),
        Summary(id="unique-id-3", video_id="test3", title="Sanallaştırma ve Bulut", summary="This is a test summary 3"),
        Summary(id="unique-id-4", video_id="test4", title="New chat", summary="This is a test summary 4"),
        Summary(id="unique-id-5", video_id="test5", title="Greeting and Inquiry", summary="This is a test summary 5"),
        Summary(id="unique-id-6", video_id="test6", title="İletişim ve Şarkı", summary="This is a test summary 6"),
    ]
    for summary in test_summaries:
        db_summary = db.query(Summary).filter(Summary.id == summary.id).first()
        if not db_summary:
            db.add(summary)
    db.commit()
    logger.info("Test data added")

# Routes
@app.get("/login")
async def login():
    flow = Flow.from_client_config(
        client_config=CLIENT_CONFIG,
        scopes=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile", "openid"],
        redirect_uri="http://localhost:5001/callback"
    )
    authorization_url, _ = flow.authorization_url(prompt="consent")
    logger.info(f"Authorization URL: {authorization_url}")
    return RedirectResponse(authorization_url)

@app.get("/callback")
async def callback(request: Request, db: Session = Depends(get_db)):
    flow = Flow.from_client_config(
        client_config=CLIENT_CONFIG,
        scopes=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile", "openid"],
        redirect_uri="http://localhost:5001/callback"
    )
    flow.fetch_token(code=request.query_params["code"])
    credentials = flow.credentials

    user_info_service = build('oauth2', 'v2', credentials=credentials)
    user_info = user_info_service.userinfo().get().execute()

    db_user = db.query(User).filter(User.google_id == user_info['id']).first()
    if not db_user:
        db_user = User(email=user_info['email'], name=user_info['name'], google_id=user_info['id'])
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

    return {"access_token": credentials.token, "refresh_token": credentials.refresh_token, "user_id": db_user.id}

@app.post("/summarize")
async def summarize_video(video: VideoURL, db: Session = Depends(get_db)):
    try:
        logger.info(f"Received summarize request for URL: {video.url}")
        video_id = get_video_id(video.url)
        if not video_id:
            logger.warning(f"Invalid YouTube URL: {video.url}")
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        existing_summary = db.query(Summary).filter(Summary.video_id == video_id).first()
        if existing_summary:
            logger.info(f"Existing summary found for video_id: {video_id}")
            return {"id": existing_summary.id, "title": existing_summary.title, "summary": existing_summary.summary}

        transcript = get_transcript(video_id)
        if not transcript:
            logger.error(f"Couldn't fetch transcript for video_id: {video_id}")
            raise HTTPException(status_code=400, detail="Couldn't fetch transcript")

        summary = summarize_text(transcript)
        if not summary:
            logger.error("Couldn't generate summary")
            raise HTTPException(status_code=500, detail="Couldn't generate summary")

        logger.info(f"Generated summary: {summary}")  # Yeni eklenen log

        title = summary.split("\n")[0]  # Assuming the first line is the title
        unique_id = str(uuid.uuid4())
        logger.info(f"Creating new summary: ID={unique_id}, Title={title}")
        new_summary = Summary(id=unique_id, video_id=video_id, title=title, summary=summary)
        db.add(new_summary)
        db.commit()
        db.refresh(new_summary)

        logger.info(f"New summary created successfully: ID={unique_id}")
        return {"id": new_summary.id, "title": new_summary.title, "summary": new_summary.summary}
    except Exception as e:
        logger.error(f"Error in summarize_video: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/summary/{unique_id}")
async def get_summary(unique_id: str, db: Session = Depends(get_db)):
    logger.info(f"Summary requested, ID: {unique_id}")
    summary = db.query(Summary).filter(Summary.id == unique_id).first()
    if not summary:
        logger.warning(f"Summary not found, ID: {unique_id}")
        raise HTTPException(status_code=404, detail=f"Summary not found: {unique_id}")
    logger.info(f"Summary found: {summary.title}")
    return {"id": summary.id, "title": summary.title, "summary": summary.summary}

@app.get("/api/summaries")
async def get_summaries(db: Session = Depends(get_db)):
    try:
        summaries = db.query(Summary).order_by(Summary.created_at.desc()).all()
        return [{"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()} for s in summaries]
    except Exception as e:
        logger.error(f"Error fetching summaries: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ask")
async def ask_question(question: Question, db: Session = Depends(get_db)):
    summary = db.query(Summary).filter(Summary.id == question.summary_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions about a summarized YouTube video. Provide concise and informative answers."},
                {"role": "user", "content": f"Based on this summary:\n\n{summary.summary}\n\nPlease answer the following question: {question.question}"}
            ]
        )
        answer = response.choices[0].message['content'].strip()
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/api/share")
async def share_summary(summary_id: str = Body(...), db: Session = Depends(get_db)):
    summary = db.query(Summary).filter(Summary.id == summary_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    # Share the summary logic here
    # For now, we'll just return a success message
    return JSONResponse({"success": True, "message": "Summary shared successfully"}, status_code=200)

@app.post("/api/rename")
async def rename_summary(summary_id: str = Body(...), new_name: str = Body(...), db: Session = Depends(get_db)):
    summary = db.query(Summary).filter(Summary.id == summary_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    summary.title = new_name
    db.commit()

    return JSONResponse({"success": True, "message": "Summary renamed successfully"}, status_code=200)

@app.post("/api/delete")
async def delete_summary(
    request: Optional[DeleteSummaryRequest] = None,
    summary_id: Optional[str] = Body(None),
    db: Session = Depends(get_db)
):
    # İstek gövdesinden veya doğrudan body'den summary_id'yi al
    if request:
        id_to_delete = request.summary_id
    elif summary_id:
        id_to_delete = summary_id
    else:
        raise HTTPException(status_code=400, detail="summary_id is required")

    summary = db.query(Summary).filter(Summary.id == id_to_delete).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    db.delete(summary)
    db.commit()
    logger.info(f"Summary deleted: ID={request.summary_id}")

    return JSONResponse({"success": True, "message": "Summary deleted successfully"}, status_code=200)

@app.post("/api/delete-all")
async def delete_all_summaries(db: Session = Depends(get_db)):
    try:
        db.query(Summary).delete()
        db.commit()
        logger.info("All summaries deleted")
        return JSONResponse({"success": True, "message": "All summaries deleted successfully"}, status_code=200)
    except Exception as e:
        logger.error(f"Error deleting all summaries: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting all summaries")

@app.post("/api/new-chat")
async def start_new_chat():
    # Start a new chat logic here
    # For now, we'll just return a new chat ID
    return JSONResponse({"chat_id": str(uuid.uuid4())}, status_code=200)

@app.get("/api/similar-sources/{summary_id}")
async def get_similar_sources(summary_id: str, db: Session = Depends(get_db)):
    summary = db.query(Summary).filter(Summary.id == summary_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    similar_sources = search_similar_sources(summary.title, api_key, cse_id)
    return similar_sources

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# Error handling middleware
@app.middleware("http")
async def errors_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Internal server error"},
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Database reset endpoint (for development purposes only)
@app.post("/reset-db")
async def reset_database():
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=403, detail="This endpoint is only available in development mode")
    
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database reset completed")
    return {"message": "Database reset successful"}

# Additional utility functions

def validate_youtube_url(url: str) -> bool:
    """Validate if the given URL is a valid YouTube URL."""
    video_id = get_video_id(url)
    return video_id is not None

async def fetch_video_details(video_id: str) -> dict:
    """Fetch video details from YouTube Data API."""
    youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()
    
    if not response['items']:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video = response['items'][0]
    return {
        "title": video['snippet']['title'],
        "description": video['snippet']['description'],
        "published_at": video['snippet']['publishedAt'],
        "view_count": video['statistics']['viewCount'],
        "like_count": video['statistics'].get('likeCount', 'N/A'),
        "duration": video['contentDetails']['duration']
    }

# Additional routes

@app.get("/video-details/{video_id}")
async def get_video_details(video_id: str):
    try:
        details = await fetch_video_details(video_id)
        return details
    except Exception as e:
        logger.error(f"Error fetching video details: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching video details")

@app.get("/user-summaries/{user_id}")
async def get_user_summaries(user_id: int, db: Session = Depends(get_db)):
    summaries = db.query(Summary).filter(Summary.user_id == user_id).order_by(Summary.created_at.desc()).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()} for s in summaries]

@app.post("/batch-summarize")
async def batch_summarize(video_urls: List[str], db: Session = Depends(get_db)):
    results = []
    for url in video_urls:
        try:
            summary = await summarize_video(VideoURL(url=url), db)
            results.append({"url": url, "success": True, "summary": summary})
        except Exception as e:
            results.append({"url": url, "success": False, "error": str(e)})
    return results

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process the received data
            # For example, you could use this to provide real-time updates on the summarization process
            await websocket.send_text(f"Processing: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

# Scheduler for periodic tasks
scheduler = BackgroundScheduler()

def cleanup_old_summaries():
    with SessionLocal() as db:
        thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        db.query(Summary).filter(Summary.created_at < thirty_days_ago).delete()
        db.commit()
        logger.info("Cleaned up old summaries")

scheduler.add_job(
    func=cleanup_old_summaries,
    trigger=IntervalTrigger(days=1),
    id='cleanup_old_summaries',
    name='Clean up summaries older than 30 days',
    replace_existing=True)

scheduler.start()

def search_similar_sources(query, api_key, cse_id):
    # Google Custom Search API için fonksiyon
    def google_search(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items'] if 'items' in res else []

    # YouTube Data API için fonksiyon
    def youtube_search(search_term, api_key, **kwargs):
        youtube = build('youtube', 'v3', developerKey=api_key)
        search_response = youtube.search().list(
            q=search_term,
            type='video',
            part='id,snippet',
            maxResults=5
        ).execute()
        return search_response.get('items', [])

    # Web siteleri arama
    websites = google_search(query, api_key, cse_id, num=5)
    
    # YouTube kanalları arama
    youtube_channels = youtube_search(query, api_key)

    results = {
        "websites": [{"title": item['title'], "link": item['link'], "type": "website"} for item in websites],
        "youtube_channels": [{"title": item['snippet']['title'], "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}", "type": "youtube"} for item in youtube_channels]
    }

    return results

@app.post("/search")
async def search_query(query: dict = Body(...)):
    try:
        query_text = query.get("query")
        if not query_text or not isinstance(query_text, str):
            raise HTTPException(status_code=400, detail="Invalid or missing query parameter")
        
        logger.info(f"Received search query: {query_text}")
        
        # Burada gerçek arama işlemlerinizi gerçekleştirin
        # Şimdilik örnek bir yanıt döndürelim
        response = {
            "sources": [
                {"title": f"Kaynak: {query_text}", "link": "https://example.com", "type": "web"},
                {"title": "YouTube Video", "link": "https://youtube.com/watch?v=123", "type": "youtube"}
            ],
            "perplexityAnswer": f"Bu '{query_text}' için bir örnek cevaptır.",
            "webSites": ["example.com", "python.org"],
            "youtubeChannels": ["Örnek Kanal 1", "Örnek Kanal 2"],
            "visualElements": [
                {
                    "imageUrl": "https://example.com/image.jpg",
                    "title": "Örnek Görsel",
                    "description": "Bu bir örnek görsel açıklamasıdır."
                }
            ]
        }
        logger.info(f"Search response generated for query: {query_text}")
        return response
    except HTTPException as he:
        logger.error(f"HTTP exception in search query: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in search query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dahili sunucu hatası: {str(e)}")

# Main function to run the application
if __name__ == "__main__":
    init_db()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001)