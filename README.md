# Advanced YouTube Video Summarizer

This project is a sophisticated web application that summarizes YouTube videos, searches for similar sources, and allows users to ask questions about the summaries.

## Features

- YouTube video summarization
- Similar source search
- Question-answering about summaries
- Summary management (delete, rename)
- User authentication with Google OAuth
- Real-time updates using WebSocket

## Technologies

- Backend: FastAPI (Python)
- Frontend: HTML, JavaScript, Tailwind CSS
- Database: PostgreSQL
- APIs: YouTube Data API, Google Custom Search API, OpenAI API

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/advanced-youtube-summarizer.git
   cd advanced-youtube-summarizer
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add the necessary API keys:
   ```
   YOUTUBE_API_KEY=your_youtube_api_key
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=postgresql://user:password@localhost/dbname
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_google_cse_id

5. Create the database:
   ```
   psql -c "CREATE DATABASE youtube_summarizer"
   ```

6. Run the application:
   ```
   uvicorn yt_sum:app --reload
   ```

7. Open `http://localhost:5001` in your browser.

## Usage

1. On the main page, enter the URL of the YouTube video you want to summarize.
2. Click the "Summarize" button and wait for the summary to be generated.
3. After the summary is created, you can view similar sources and ask questions about the summary.
4. In the sidebar on the left, you can view your previous summaries, delete them, or rename them.
5. Switch to search mode to search for websites and YouTube videos on a specific topic.

## API Endpoints

- `POST /summarize`: Summarizes a YouTube video
- `GET /summary/{unique_id}`: Retrieves details of a specific summary
- `POST /ask`: Asks a question about the summary
- `POST /search`: Searches for websites and YouTube videos
- `POST /api/delete`: Deletes a summary
- `POST /api/delete-all`: Deletes all summaries
- `POST /api/rename`: Renames a summary
- `GET /api/similar-sources/{summary_id}`: Retrieves similar sources

## Development

- `yt_sum.py`: Main FastAPI application and backend logic
- `static/index.html`: Frontend interface and JavaScript code## Contributing

1. Fork this repository
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

Project Owner: [Your Name] - email@example.com

Project Link: [https://github.com/your-username/advanced-youtube-summarizer](https://github.com/your-username/advanced-youtube-summarizer)


