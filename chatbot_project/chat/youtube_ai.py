import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import google.generativeai as genai

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    else:
        return None

def check_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
            transcript_data = transcript.fetch()
            transcript_text = " ".join([entry['text'] for entry in transcript_data])
            return transcript_text, 'en'
        except NoTranscriptFound:
            for transcript in transcript_list:
                transcript_data = transcript.fetch()
                transcript_text = " ".join([entry['text'] for entry in transcript_data])
                return transcript_text, transcript.language_code
    except (TranscriptsDisabled, NoTranscriptFound):
        return None, None
    except Exception as e:
        print(f"An error occurred while fetching transcript: {e}")
        return None, None

def generate_video_summary(captions, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"You create comprehensive and accurate summaries from captions of a video, explaining concepts and keeping the overall tone of the video. The captions of the video are: {captions}. Please summarise."

    response = model.generate_content(prompt)
    return response.text.strip()

def question_video_summary(summary, question, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"You answer questions on video summaries from a user and give comprehensive answers drawing on your wider knowledge where needed. Please answer this {question} based on this summary {summary}."

    response = model.generate_content(prompt)
    return response.text.strip()
