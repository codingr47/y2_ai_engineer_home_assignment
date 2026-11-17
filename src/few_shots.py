"""Reviewed, fixed training demonstrations; serving never imports training code."""
import hashlib
import json
from pathlib import Path

from src.openai_prompt import EXTRACTOR_SYSTEM_TEMPLATE, IMPORTANT_GUIDELINES

EXAMPLES = Path(__file__).with_name("few_shot_examples.json")


def prompt_configuration(catalog, count=0, *, guidelines_enabled=False):
    if type(count) is not int or count not in {0, 6, 12}:
        raise ValueError("few-shot count must be 0, 6, or 12")
    source = EXAMPLES.read_bytes()
    artifact = json.loads(source)
    examples = artifact["examples"][:count]
    if len(examples) != count or len({example["source_id"] for example in examples}) != count:
        raise ValueError("incomplete or duplicate few-shot examples")
    demonstrations = []
    for example in examples:
        if example["review_status"] != "reviewed" or example["source_split"] != "train":
            raise ValueError("few-shot examples must be reviewed training rows")
        catalog.validate({**example["expected"], "confidence": .5})
        demonstrations.append("QUERY: " + json.dumps(example["query"], ensure_ascii=False)
                              + "\nEXPECTED EXTRACTION: " + json.dumps(example["expected"], ensure_ascii=False))
    prompt = EXTRACTOR_SYSTEM_TEMPLATE.format(catalog=catalog.prompt())
    if demonstrations:
        prefix = ("REVIEWED EXTRACTION EXAMPLES\nThese trusted examples demonstrate category and params only. "
                  "Your response must also include your own confidence as required above. "
                  "Do not copy a confidence rating from an example.\n\n" + "\n\n".join(demonstrations) + "\n\n")
        # Keep the supplied catalog text byte-for-byte identical and in the same final position.
        marker = "CATALOG\n"
        before, catalog_text = prompt.split(marker, 1)
        prompt = before + prefix + marker + catalog_text
    if guidelines_enabled:
        before, catalog_text = prompt.split("CATALOG\n", 1)
        prompt = before + IMPORTANT_GUIDELINES + "\nCATALOG\n" + catalog_text
    metadata = {"few_shot_count": count, "examples_version": artifact["version"],
                "guidelines_enabled": guidelines_enabled,
                "examples_sha256": hashlib.sha256(source).hexdigest(),
                "example_ids": [example["source_id"] for example in examples],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "taxonomy_sha256": hashlib.sha256(catalog.source).hexdigest()}
    return prompt, metadata
