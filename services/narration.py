"""내레이션 텍스트 정리 — TTS로 읽을 '순수한 대사'만 남긴다.

음성 엔진은 받은 텍스트를 그대로 읽으므로, 대본에 섞여 들어온 연출 지시가 소리로 나오지 않도록 제거한다.
(TTS 스타일 지시는 tts_service에서 텍스트와 분리된 별도 필드로 전달 — 여기서는 대본 쪽 정리만 담당)

제거 대상
- 대괄호 지시문: [밝게], [효과음: 박수], 【BGM】 → 항상 제거 (대본에서 대괄호는 사실상 지시문)
- 괄호 지시문: (웃으며), (빠른 템포로), (잠시 멈춤) → 연출 단어가 있거나 '~게/~며/~듯'으로 끝나는 짧은 괄호만.
  (2024), (원제: …) 같은 내용 괄호는 남긴다
- 화자 표기: 줄 맨 앞의 "내레이션:", "JB:", "성우:" …
- 지시 문장: "다음 문장을 밝은 톤으로 읽어줘:" 처럼 콜론으로 끝나는 읽기 지시, 설정된 스타일 지시문 자체
- 마크다운·목록 기호, 꺾쇠 태그(<pause>), 이모지, 대본 전체를 감싼 따옴표
"""

import re

DIRECTION_WORDS = (
    "톤", "템포", "속도", "빠르게", "천천히", "느리게", "밝게", "어둡게", "차분", "웃으며", "웃음", "미소", "속삭",
    "강조", "효과음", "배경음", "음악", "BGM", "SFX", "쉼", "멈춤", "정적", "내레이션", "나레이션", "목소리", "말투",
    "한숨", "숨을", "감정", "읽", "낭독", "발음", "억양", "톤으로", "리듬",
    "tone", "tempo", "whisper", "laugh", "pause", "slowly", "quickly", "narrat", "voice", "sfx", "bgm",
)
SPEAKER_LABELS = ("내레이션", "나레이션", "내레이터", "성우", "진행자", "화자", "해설", "자막", "대사", "NARRATOR",
                  "Narrator", "narrator", "VO", "V.O.", "JB", "N")
READ_INSTRUCTION = re.compile(r"(읽어|말해|낭독|발음|톤|템포|스타일|목소리|read|say|speak)", re.IGNORECASE)

_BRACKETS = re.compile(r"\[[^\]]*\]|【[^】]*】|〔[^〕]*〕")
_PARENS = re.compile(r"[(（]([^()（）]{0,40})[)）]")
_ANGLE_TAGS = re.compile(r"<[^<>]{0,40}>")
_SPEAKER = re.compile(r"^\s*(?:" + "|".join(re.escape(s) for s in SPEAKER_LABELS) + r")\s*[:：]\s*", re.MULTILINE)
_MARKDOWN = re.compile(r"\*\*|__|`|^\s*#+\s*|^\s*[-*•·]\s+", re.MULTILINE)


def _is_direction(inner: str) -> bool:
    inner = inner.strip()
    if not inner:
        return True
    if any(w.lower() in inner.lower() for w in DIRECTION_WORDS):
        return True
    # (힘주어 말하듯), (조용하게), (웃으며) 처럼 짧은 부사형
    return len(inner) <= 12 and bool(re.search(r"(게|며|듯|면서|하게|히)$", inner))


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF
            or cp in (0xFE0F, 0xFE0E, 0x200D, 0x20E3))


def sanitize_narration(text: str, instructions: tuple[str, ...] = ()) -> str:
    """연출 지시·표기를 걷어낸 낭독용 텍스트. instructions: 혹시 섞여 들어왔을 수 있는 지시문(예: TTS 스타일)."""
    s = (text or "").replace("\r\n", "\n")
    for inst in instructions:
        inst = (inst or "").strip()
        if inst:
            s = re.sub(re.escape(inst) + r"\s*[:：]?", " ", s)
    s = _SPEAKER.sub("", s)
    # 콜론으로 끝나는 읽기 지시 줄/문장 ("빠른 템포로 읽어줘:")
    s = re.sub(r"(^|\n)[^\n:：]{0,80}[:：]\s*(?=\n|$)",
               lambda m: m.group(1) if READ_INSTRUCTION.search(m.group(0)) else m.group(0), s)
    s = re.sub(r"(^|\n)\s*[^\n:：]{0,60}(읽어\s*줘|읽어\s*주세요|말해\s*줘|말해\s*주세요)[^\n:：]{0,20}[:：]\s*",
               lambda m: m.group(1), s)
    s = _BRACKETS.sub(" ", s)
    s = _PARENS.sub(lambda m: " " if _is_direction(m.group(1)) else m.group(0), s)
    s = _ANGLE_TAGS.sub(" ", s)
    s = _MARKDOWN.sub("", s)
    s = "".join(ch for ch in s if not _is_emoji(ch))
    s = re.sub(r"\s+", " ", s).strip()
    # 대본 전체를 감싼 따옴표
    pairs = {'"': '"', "“": "”", "'": "'", "‘": "’", "「": "」", "『": "』"}
    if len(s) >= 2 and s[0] in pairs and s[-1] == pairs[s[0]] and s.count(s[0]) == 1 + (s[0] == s[-1]):
        s = s[1:-1].strip()
    # 지시문을 지운 자리에 남은 문장부호 정리
    s = re.sub(r"\s+([,.!?…])", r"\1", s)
    s = re.sub(r"^[,.:：\s]+", "", s)
    return s
