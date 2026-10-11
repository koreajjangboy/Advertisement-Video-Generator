"""예전 방식으로 만든 Luma 클립 식별용 (고정 사본 — 수정 금지).

v1 (포즈 이미지를 첫 프레임으로, 영상 프롬프트를 'Scene atmosphere:'로 중간에 끼워 넣던 시기)의 프롬프트·키 계산을
그대로 보존해, 이미 비용을 들여 만든 클립이 어떤 대본·설정으로 만들어졌는지 확인한다 (clip_service.adopt_legacy_clips).
"""

import hashlib
import json

from config import LUMA_CHARACTER, LUMA_RESOLUTION

_ACTING_V1 = {
    "greeting": "smiles warmly at the camera and gives a friendly wave hello",
    "question": "tilts his head with a curious, questioning look, eyebrows slightly raised, as if asking the viewer",
    "surprise": "gasps in genuine surprise, eyes widening, hand rising to his chest",
    "emphasis": "nods with conviction and points toward the camera with a bright, confident smile",
    "reflection": "looks away thoughtfully with his chin resting on his hand, lost in a quiet memory",
    "conclusion": "leans slightly toward the camera and points forward, warmly inviting the viewer",
}
_CAMERA_V1 = {
    "hook": "slow dramatic push-in toward his face",
    "body": "gentle handheld dolly with soft parallax",
    "cta": "slow pull-back revealing the scene",
}


def prompt_v1(section: str, visual_prompt: str, acting) -> str:
    actions = [_ACTING_V1.get(b.emotion, _ACTING_V1["emphasis"]) for b in acting.beats]
    action = actions[0] + (f", then {actions[1]}" if len(actions) > 1 else "")
    mood = " ".join((visual_prompt or "").split())[:500]
    parts = [
        f"Cinematic vertical film scene. {LUMA_CHARACTER[:1].upper()}{LUMA_CHARACTER[1:]} from the reference image "
        f"{action}.",
        f"Scene atmosphere: {mood}." if mood else "",
        f"Camera: {_CAMERA_V1.get(section, _CAMERA_V1['body'])}, shallow depth of field, soft cinematic lighting, "
        "film grain.",
        "Natural, expressive facial acting and subtle body motion; keep the same face, hairstyle and outfit "
        "throughout. No text, no captions, no logos.",
    ]
    return " ".join(p for p in parts if p)[:1200]


def content_key_v1(prompt: str, pose_signature: list) -> str:
    raw = json.dumps([prompt, *pose_signature], ensure_ascii=False)
    return "luma_c_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def file_keys_v1(prompt: str, pose_signature: list) -> set[str]:
    """v1 캐시 파일 이름 후보 (API 종류·모델·길이 조합)."""
    keys = set()
    for flavor, model in (("agents", "ray-3.2"), ("dream-machine", "ray-2")):
        for duration in ("5s", "9s"):
            raw = json.dumps([flavor, model, LUMA_RESOLUTION, duration, "9:16", prompt, *pose_signature],
                             ensure_ascii=False)
            keys.add("luma_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20])
    return keys
