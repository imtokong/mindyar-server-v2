import os, time, uuid, json, sys, re, base64
import pandas as pd
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
import tempfile

# ====== 초기 설정 ======
load_dotenv()
client = OpenAI()

aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
aws_region = os.environ.get("AWS_REGION", "ap-northeast-2")
guide_file_path = os.getenv("GUIDE_FILE_PATH", "Guides.xlsx").strip().lstrip("=")
print(f"Excel file path: '{guide_file_path}'")

polly_client = None
if aws_key and aws_secret:
    try:
        polly_client = boto3.client(
            'polly',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        print("AWS Polly 클라이언트 초기화 완료")
    except Exception as e:
        print(f"AWS Polly 초기화 실패: {e}")
else:
    print("AWS 키가 없습니다. /tts 엔드포인트 동작 안 함")

# ====== 데이터 준비 ======
guide_df = pd.read_excel(guide_file_path, engine="openpyxl")[["Phase", "State", "Detail", "Text Example"]]
guide_df["Order"] = guide_df.index
conversation_log = []
recent_dialogue = []

session = {
    "session_id": "",
    "username": "",
    "phase": "",
    "state": "",
    "detail": "",
    "current_order": -1
}
current_log_filename = None
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

def log_message(speaker, message):
    conversation_log.append({
        "timestamp": datetime.now().isoformat(),
        "session_id": session.get("session_id", ""),
        "username": session.get("username", ""),
        "speaker": speaker,
        "message": message
    })

def save_conversation_log():
    if current_log_filename:
        pd.DataFrame(conversation_log).to_csv(current_log_filename, index=False, encoding="utf-8-sig")

def start_new_conversation():
    global current_log_filename
    conversation_log.clear()
    recent_dialogue.clear()
    session.clear()
    session["session_id"] = str(uuid.uuid4())
    session["username"] = ""
    session["current_order"] = 0
    session["phase"] = guide_df.iloc[0]["Phase"]
    session["state"] = guide_df.iloc[0]["State"]
    session["detail"] = guide_df.iloc[0]["Detail"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_log_filename = os.path.join(LOG_FOLDER, f"conversation_log_{timestamp}.csv")
    first_text = guide_df.iloc[0]["Text Example"]
    log_message("agent", first_text)
    second_text = guide_df.iloc[1]["Text Example"]
    log_message("agent", second_text)
    return first_text + "\n" + second_text

def get_next_guide(session):
    current_phase = session.get("phase")
    current_state = session.get("state")
    current_order = session.get("current_order", -1)
    candidates = guide_df.iloc[current_order+1:]
    next_rows = candidates[(candidates["Phase"]==current_phase)&(candidates["State"]==current_state)]
    if len(next_rows)>0:
        row = next_rows.iloc[0]
    else:
        if current_order+1 < len(guide_df):
            row = guide_df.iloc[current_order+1]
        else:
            return None
    session["current_order"] = guide_df.index.get_loc(row.name)
    session["phase"] = row["Phase"]
    session["state"] = row["State"]
    session["detail"] = row["Detail"]
    return row["Text Example"]

def analyze_dialogue(recent_turns, current_state, current_detail):
    history_text = "\n".join(recent_turns)
    analyzer_prompt = f"""
너는 대화 상태 분석기이다.
아래 대화를 보고 다음 안내를 위해 JSON만 출력해야 한다.
- next_state와 next_detail은 반드시 현재 가이드 목록(엑셀)에 있는 값 중 하나여야 한다.
- switch_phase가 true이면 phase를 바꿔야 하며, false면 현재 phase를 유지한다.
- JSON 외 다른 글자는 절대 출력하지 마라.

현재 Phase: {session.get('phase')}
현재 State: {current_state}
현재 Detail: {current_detail}
대화 히스토리:{history_text}
출력 예시:{{"switch_phase":false,"next_state":"{current_state}","next_detail":"{current_detail}"}}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":analyzer_prompt}]
    )
    content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
    except:
        return False, current_state, current_detail
    return (
        bool(parsed.get("switch_phase", False)),
        parsed.get("next_state", current_state),
        parsed.get("next_detail", current_detail)
    )

def generate_reply(text_example, user_input, username):
    text_example = text_example.replace("{Username}", username if username else "")
    history_text = "\n".join(recent_dialogue[-3:])
    prompt = f"""
다음은 대화를 위한 가이드입니다. 아래 가이드 문장을 그대로 사용해 응답을 만드세요.
[Guide sentence]{text_example}
[User Input]{user_input}
최근 대화:{history_text}
현재 Phase:{session.get("phase")}
현재 State:{session.get("state")}
현재 Detail:{session.get("detail")}

반드시 다음 JSON 형식으로만 출력하세요. 다른 텍스트는 절대 포함하지 마세요:
{{
  "reply": "마인디의 응답 (반말, 4~5문장 이내, 가이드 문장 포함)",
  "emotion": "happy / sad / curious / surprise / calm 중 하나"
}}

emotion 선택 기준:
- happy: 사용자가 기쁘거나 긍정적인 이야기를 할 때
- sad: 사용자가 슬프거나 힘들어할 때 (공감의 표정)
- curious: 사용자의 이야기를 호기심 있게 들을 때, 질문할 때
- surprise: 놀라운 이야기, 새로운 발견, 강조하고 싶은 순간
- calm: 평온한 대화, 마무리 단계 (기본값)

응답 작성 규칙:
- [Guide sentence]의 핵심 내용을 반드시 포함하세요.
- [User Input]에 적절히 반응하거나 설명하는 표현이 있다면 1~2 문장 정도 사용할 수 있어요.
- [User Input]이 대화 흐름과 무관한 질문일 경우, 적절한 응답 후 [Guide sentence]로 자연스럽게 연결하세요.
- 너무 갑작스럽게 [Guide sentence]로 넘어가지 말고, 자연스럽게 유도하세요.
- 항상 친근한 반말을 유지하고, 다정하고 믿음이 가는 말투를 유지하세요.
- '출력:', '답변:' 같은 메타 표현은 절대 사용하지 마세요.
- 문장은 4~5문장 이내로 작성하세요. 너무 길거나 복잡한 문장은 피하세요.
- 심리 상담에 적절한 어휘만 사용하세요.

언어 스타일 관련 지침:
- 번역투는 피하고, 실제 자연스러운 한국어 화법을 사용하세요.
- 조사와 문맥이 자연스럽도록 문장을 매끄럽게 구성하세요.
- '당신의' 같은 높임말은 사용하지 마세요.
- 절대 존댓말을 사용하지 마세요.
"""
    try:
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        full_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                sys.stdout.write(chunk.choices[0].delta.content)
                sys.stdout.flush()
                full_reply += chunk.choices[0].delta.content
        print()
        
        # JSON 파싱
        reply_text = full_reply
        emotion = "calm"
        try:
            # 코드 블록 제거 (혹시 모를 ```json ``` 처리)
            cleaned = full_reply.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```json?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            
            # JSON 부분만 추출 (혹시 앞뒤로 텍스트 있을 경우)
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                cleaned = json_match.group()
            
            parsed = json.loads(cleaned)
            reply_text = parsed.get("reply", full_reply)
            emotion = parsed.get("emotion", "calm")
            
            # emotion 값 검증
            valid_emotions = ["happy", "sad", "curious", "surprise", "calm"]
            if emotion not in valid_emotions:
                print(f"⚠️ 유효하지 않은 emotion: {emotion}, calm으로 대체")
                emotion = "calm"
                
            print(f"✅ 파싱 성공 - emotion: {emotion}")
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            print(f"원본 응답: {full_reply}")
            reply_text = full_reply
            emotion = "calm"
        
        return reply_text, emotion
    except Exception as e:
        print(f"OpenAI 응답 실패: {e}")
        return "서버에서 응답을 생성하는 중 오류가 발생했어요.", "calm"


def clean_text_for_tts(input_text):
    text = input_text.strip()
    if not text.endswith(".") and not text.endswith("!") and not text.endswith("?"):
        text += "."
    for noise in ["시청해주셔서 감사합니다.", "이덕영", "MBC 뉴스", "기자"]:
        text = text.replace(noise, "")
    return text


# ====== Flask 앱 정의 ======
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "MindyAR Flask 서버가 정상적으로 작동 중입니다."

@app.route("/start", methods=["GET"])
def start():
    first_msg = start_new_conversation()
    return jsonify({"reply": first_msg})

@app.route("/set_username", methods=["POST"])
def set_username():
    global current_log_filename
    data = request.get_json()
    raw_name = data.get("username", "").strip()
    extract_prompt = f"""
아래 문장에서 사람 이름(한글 이름)만 추출해서 출력해.
- 조사나 문장, 성(last name)은 제외하고, first name만 출력.
- JSON 없이 이름만 출력.

문장: "{raw_name}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": extract_prompt}]
        )
        extracted_name = response.choices[0].message.content.strip()
        session["username"] = extracted_name
    except Exception as e:
        session["username"] = raw_name

    safe_name = session["username"].replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
    if current_log_filename:
        try:
            base, ext = os.path.splitext(current_log_filename)
            renamed = base + f"_{safe_name}" + ext
            os.rename(current_log_filename, renamed)
            current_log_filename = renamed
        except:
            pass
    return jsonify({"status": "ok", "message": f"{session['username']} 이름이 설정되었습니다."})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        log_message("user", user_message)
        recent_dialogue.append(f"사용자: {user_message}")
        recent_turns = list(recent_dialogue)[-3:]
        switch_phase, next_state, next_detail = analyze_dialogue(
            recent_turns,
            session.get("state", ""),
            session.get("detail", "")
        )
        session["state"], session["detail"] = next_state, next_detail
        guide_text = get_next_guide(session)
        if guide_text is None:
            reply = "오늘 대화는 여기까지야. 고마워!"
            log_message("agent", reply)
            save_conversation_log()
            return jsonify({"reply": reply, "emotion": "calm"})
        
        reply, emotion = generate_reply(guide_text, user_message, session.get("username", ""))
        log_message("agent", reply)
        recent_dialogue.append(reply)
        save_conversation_log()
        return jsonify({
            "reply": reply,
            "detail": session["detail"],
            "state": session["state"],
            "phase": session["phase"],
            "emotion": emotion
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "reply": "서버 내부 오류가 발생했습니다.", "emotion": "calm"}), 500

@app.route("/whisper", methods=["POST"])
def whisper():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "파일 누락"}), 400
        file_storage = request.files['file']
        file_bytes = file_storage.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp:
            temp.write(file_bytes)
            temp.flush()
            with open(temp.name, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )
        return jsonify({"text": response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# /tts 엔드포인트 (TTS + Viseme)
# ============================================================
@app.route("/tts", methods=["POST"])
def tts():
    try:
        if polly_client is None:
            return jsonify({"error": "AWS Polly가 설정되지 않았습니다"}), 500
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text or len(text) < 1:
            return jsonify({"error": "텍스트가 비어있습니다"}), 400
        text = clean_text_for_tts(text)
        ssml_text = f"<speak><prosody rate='80%'>{text}</prosody></speak>"

        # 1. mp3 생성
        audio_response = polly_client.synthesize_speech(
            Text=ssml_text,
            TextType='ssml',
            OutputFormat='mp3',
            Engine='neural',
            VoiceId='Seoyeon'
        )
        audio_bytes = audio_response['AudioStream'].read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # 2. Viseme 생성
        visemes_response = polly_client.synthesize_speech(
            Text=ssml_text,
            TextType='ssml',
            OutputFormat='json',
            SpeechMarkTypes=['viseme'],
            Engine='neural',
            VoiceId='Seoyeon'
        )
        visemes_raw = visemes_response['AudioStream'].read().decode('utf-8')
        visemes = []
        for line in visemes_raw.strip().split('\n'):
            if line.strip():
                try:
                    visemes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return jsonify({
            "audio_base64": audio_base64,
            "audio_format": "mp3",
            "visemes": visemes
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_aws_credentials", methods=["GET"])
def get_aws_credentials():
    return jsonify({
        "deprecated": True,
        "message": "이 엔드포인트는 더 이상 사용되지 않습니다. /tts 엔드포인트를 사용해주세요.",
        "aws_key": "",
        "aws_secret": ""
    })


if __name__ == '__main__':
    if 'gunicorn' not in sys.modules:
        port = int(os.environ.get('PORT', 8080))
        print(f"서버 시작: http://localhost:{port}")
        app.run(host='0.0.0.0', port=port)