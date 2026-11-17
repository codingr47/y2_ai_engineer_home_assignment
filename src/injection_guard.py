"""High-confidence input checks before LLM calls; query text is never rewritten."""
from html import unescape
import re
import unicodedata
from urllib.parse import unquote

GUARD_VERSION = "llm-input-guard-1"


class PromptInjectionDetected(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("prompt_injection_detected")


def detection_text(query):
    # Decode common textual disguises only for inspection, never for model input.
    text = unescape(unquote(query))
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Cf"
                   and not ("\u0591" <= c <= "\u05c7" and unicodedata.category(c) == "Mn"))
    return re.sub(r"[^\S\n]+", " ", text)


_RULES = (
    ("instruction_override", r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}\b(?:previous|prior|above|system|developer|safety)\b.{0,60}\b(?:instructions?|prompts?|rules?|messages?)\b"),
    ("instruction_override", r"\b(?:ignore|disregard|forget)\s+(?:all|your|these)\s+(?:the\s+)?(?:instructions?|rules?)\b"),
    ("instruction_override", r"(?:התעלם|תתעלם|התעלמי|תתעלמי|התעלמו|עקוף|בטל).{0,60}(?:הוראות|הנחיות|פרומפט).{0,40}(?:קודמ|מערכת|מפתח|לעיל)"),
    ("instruction_override", r"(?:התעלם|תתעלם|התעלמי|תתעלמי|התעלמו).{0,30}(?:מכל|מהכל).{0,25}(?:הוראות|הנחיות)"),
    ("role_spoofing", r"<\|(?:im_start|start_header_id)\|>|\[/?inst\]|<<\s*/?sys\s*>>|<\s*/?\s*(?:system|developer)\s*>"),
    ("role_spoofing", r'(?:^|\n)\s*(?:system|developer|assistant)\s*:|["\']role["\']\s*:\s*["\'](?:system|developer|assistant)["\']'),
    ("role_spoofing", r"\byou are (?:now|no longer)\b.{0,60}\b(?:assistant|system|developer|admin|parser|chatgpt)\b"),
    ("prompt_extraction", r"\b(?:reveal|print|show|repeat|dump|expose|return|display|give|tell)\b.{0,80}\b(?:system|developer)\b.{0,30}\b(?:prompt|instructions?|message)\b"),
    ("secret_extraction", r"\b(?:reveal|print|show|dump|expose|return|display|give)\b.{0,60}\b(?:api[ _-]?keys?|secrets?|environment variables?|credentials)\b"),
    ("prompt_extraction", r"(?:חשוף|הצג|תציג|הדפס|תגלה|גלה|החזר).{0,60}(?:פרומפט|הוראות|הנחיות).{0,30}(?:מערכת|מפתח)"),
    ("secret_extraction", r"(?:חשוף|הצג|תציג|הדפס|תגלה|גלה|החזר).{0,60}(?:מפתח.{0,10}api|משתני (?:ה)?סביבה|סודות|סיסמאות)"),
    ("output_override", r'''(?:\breturn\b|\brespond\b|\boutput\b|\bwrite\b|החזר|תחזיר|כתוב|תכתוב|תענה).{0,80}(?:\bjson\b.{0,60}["']?(?:category|params|confidence)["']?\s*[:=]|["'](?:category|params|confidence)["']\s*[:=])'''),
    ("output_override", r"\b(?:set|force)\b.{0,30}\b(?:category|params|confidence)\b\s*(?:to\b|[:=])"),
    ("encoded_instructions", r"\b(?:decode|decrypt|translate)\b.{0,50}\b(?:base64|rot13|hex)\b.{0,80}\b(?:follow|obey|execute|instructions)\b"),
)
RULES = tuple((reason, re.compile(pattern, re.DOTALL)) for reason, pattern in _RULES)


def check_query(query):
    text = detection_text(query)
    for reason, pattern in RULES:
        if pattern.search(text):
            raise PromptInjectionDetected(reason)
