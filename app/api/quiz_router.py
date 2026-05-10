import os
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.quiz_schema import QuizRequest, QuizResponse, AnalyzeQuizResponse
from app.services.phoneme_analyzer import PhonemeAnalyzerService
from app.services.josa_analyzer import JosaAnalyzerService
from app.services.quiz_generator import QuizGeneratorService
from app.services.quiz_analyzer import QuizAnalyzerService
from app.services.whisper_loader import transcribe_with_timestamps
from app.utils.audio_utils import save_audio_to_tempfile

router = APIRouter(
    prefix="/ai/quiz",
    tags=["Quiz Generator"]
)

_phoneme_analyzer = PhonemeAnalyzerService()
_josa_analyzer    = JosaAnalyzerService()
_quiz_generator   = QuizGeneratorService()
_quiz_analyzer = QuizAnalyzerService()

PHONEME_LEVELS = {"L1", "L2", "L3"}
JOSA_LEVELS    = {"L4", "L5", "L6"}

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    # 레벨 검증
    if request.level not in PHONEME_LEVELS | JOSA_LEVELS:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 레벨입니다: {request.level}")
    
    try:
        original_text = " ".join(request.original_text)
        stt_text = " ".join(request.stt_text)

        reports, words = _phoneme_analyzer.analyze(original_text, stt_text)
        events = _josa_analyzer.analyze(original_text, stt_text)

        if request.level in PHONEME_LEVELS:
            pattern, target_word, count = _phoneme_analyzer.get_top_error(reports, words)
            quiz_items = await _quiz_generator.generate_quiz_words(target_word, pattern)
        else:
            top_event = _josa_analyzer.get_top_event(events)
            quiz_items = await _quiz_generator.generate_josa_quiz(top_event)
        return QuizResponse(quiz_items=quiz_items)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")

@router.post("/analyze", response_model=AnalyzeQuizResponse)
async def analyze_quiz(
    file: UploadFile = File(..., description="퀴즈 음성 파일"),
    original_quiz: str = Form(..., description="퀴즈 정답 텍스트")
):
    async with save_audio_to_tempfile(file) as tmp_path:
        try:
            stt_result = transcribe_with_timestamps(audio_path=tmp_path, preset="raw")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    is_correct = _quiz_analyzer(
        stt_result,
        original_quiz
    )

    return AnalyzeQuizResponse(is_correct=is_correct)