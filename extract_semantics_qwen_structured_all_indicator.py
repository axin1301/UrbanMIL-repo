import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_IMG = "/pfs/lustrep4/scratch/project_462001163/xiyanxin/UrbanICL/dataset/Citylens_images"


def fix_path(path: str) -> str:
    return os.path.join(BASE_IMG, os.path.basename(path))


def empty_record(region_id: str, sample_index: int | None = None, satellite_image_name: str = "", source_region_id: str = "") -> dict:
    return {
        "region_id": region_id,
        "sample_index": sample_index if sample_index is not None else -1,
        "source_region_id": source_region_id,
        "satellite_image_name": satellite_image_name,
        "urban_form": {
            "building_density": 0.0,
            "vertical_density": 0.0,
            "road_density": 0.0,
            "impervious_surface": 0.0,
            "block_compactness": 0.0,
            "urban_sprawl": 0.0,
        },
        "economic_function": {
            "commercial_intensity": 0.0,
            "industrial_intensity": 0.0,
            "business_activity": 0.0,
            "night_economy_potential": 0.0,
            "logistics_activity": 0.0,
            "land_use_mix": 0.0,
        },
        "livability": {
            "residential_quality": 0.0,
            "building_quality": 0.0,
            "green_amenity": 0.0,
            "walkability": 0.0,
            "street_cleanliness": 0.0,
            "public_service_access": 0.0,
        },
        "mobility": {
            "transport_accessibility": 0.0,
            "road_hierarchy": 0.0,
            "parking_intensity": 0.0,
            "traffic_activity": 0.0,
            "connectivity": 0.0,
            "transit_oriented_development": 0.0,
        },
        "socioeconomic_prior": {
            "income_level_prior": 0.0,
            "urban_development_level": 0.0,
            "informality_probability": 0.0,
            "energy_consumption_intensity": 0.0,
            "construction_activity": 0.0,
            "urban_maturity": 0.0,
        },
        "uncertainty": 0.0,
    }


def build_model():
    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            # torch_dtype="auto",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            # torch_dtype="auto",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def build_messages(image_paths: list[str]):
    content = [{"type": "image", "image": f"file://{path}"} for path in image_paths]
    prompt = """
You are an urban analytics expert.

Given 1 satellite image and 10 street-view images from the same urban region, estimate a structured urban profile.

Return JSON only, with exactly this schema and no extra text:
{
  "region_id": "...",
  "urban_form": {
    "building_density": 0.0,
    "vertical_density": 0.0,
    "road_density": 0.0,
    "impervious_surface": 0.0,
    "block_compactness": 0.0,
    "urban_sprawl": 0.0
  },
  "economic_function": {
    "commercial_intensity": 0.0,
    "industrial_intensity": 0.0,
    "business_activity": 0.0,
    "night_economy_potential": 0.0,
    "logistics_activity": 0.0,
    "land_use_mix": 0.0
  },
  "livability": {
    "residential_quality": 0.0,
    "building_quality": 0.0,
    "green_amenity": 0.0,
    "walkability": 0.0,
    "street_cleanliness": 0.0,
    "public_service_access": 0.0
  },
  "mobility": {
    "transport_accessibility": 0.0,
    "road_hierarchy": 0.0,
    "parking_intensity": 0.0,
    "traffic_activity": 0.0,
    "connectivity": 0.0,
    "transit_oriented_development": 0.0
  },
  "socioeconomic_prior": {
    "income_level_prior": 0.0,
    "urban_development_level": 0.0,
    "informality_probability": 0.0,
    "energy_consumption_intensity": 0.0,
    "construction_activity": 0.0,
    "urban_maturity": 0.0
  },
  "uncertainty": 0.0
}

Rules:
- Every numeric value must be a float between 0 and 1.
- Use 0.0 only if evidence is extremely weak; otherwise provide a calibrated score.
- Higher means stronger / more likely / better developed.
- Keep region_id exactly as provided.
- Do not add explanation, markdown, or extra fields.
""".strip()
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_json_substring(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_float(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not torch.isfinite(torch.tensor(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalize_record(region_id: str, obj: dict | None, sample_index: int, satellite_image_name: str, source_region_id: str) -> dict:
    out = empty_record(region_id, sample_index=sample_index, satellite_image_name=satellite_image_name, source_region_id=source_region_id)
    if not isinstance(obj, dict):
        return out
    out["region_id"] = region_id
    out["sample_index"] = int(sample_index)
    out["source_region_id"] = str(source_region_id)
    out["satellite_image_name"] = str(satellite_image_name)
    for group_name in ["urban_form", "economic_function", "livability", "mobility", "socioeconomic_prior"]:
        group = obj.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for key in out[group_name]:
            out[group_name][key] = _coerce_float(group.get(key, out[group_name][key]))
    out["uncertainty"] = _coerce_float(obj.get("uncertainty", 0.0))
    return out


def postprocess_output(text: str, region_id: str, sample_index: int, satellite_image_name: str, source_region_id: str) -> dict:
    candidate = (text or "").strip()
    if not candidate:
        return empty_record(region_id, sample_index=sample_index, satellite_image_name=satellite_image_name, source_region_id=source_region_id)
    if not (candidate.startswith("{") and candidate.endswith("}")):
        candidate = _extract_json_substring(candidate) or ""
    try:
        obj = json.loads(candidate)
    except Exception:
        return empty_record(region_id, sample_index=sample_index, satellite_image_name=satellite_image_name, source_region_id=source_region_id)
    return normalize_record(region_id, obj, sample_index=sample_index, satellite_image_name=satellite_image_name, source_region_id=source_region_id)


@torch.no_grad()
def infer_one(
    model,
    processor,
    image_paths: list[str],
    region_id: str,
    sample_index: int,
    satellite_image_name: str,
    source_region_id: str,
    max_new_tokens: int = 256,
) -> dict:
    messages = build_messages(image_paths)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return postprocess_output(
        output_text,
        region_id,
        sample_index=sample_index,
        satellite_image_name=satellite_image_name,
        source_region_id=source_region_id,
    )


def load_existing_index(out_path: str) -> dict[str, dict]:
    if not os.path.exists(out_path):
        return {}
    with open(out_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return {}
    if not isinstance(data, list):
        return {}
    out = {}
    for row in data:
        if isinstance(row, dict) and "region_id" in row:
            out[str(row["region_id"])] = row
    return out


def save_records(path: str, records: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main(
    *,
    json_path: str,
    out_json: str,
    start_idx: int = 0,
    end_idx=None,
    max_new_tokens: int = 256,
    resume: bool = False,
    flush_every: int = 10,
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if end_idx is None:
        end_idx = len(data)

    existing = load_existing_index(out_json) if resume else {}
    ordered_records = list(existing.values()) if resume else []
    seen = {str(row["region_id"]) for row in ordered_records if isinstance(row, dict) and "region_id" in row}

    model = None
    processor = None
    flush_every = max(1, int(flush_every))

    for idx in range(start_idx, end_idx):
        item = data[idx]
        source_region_id = str(item.get("region_id", item.get("area", "")))
        region_id = f"sample_{idx:06d}"
        if resume and region_id in existing:
            continue
        image_paths = [fix_path(path) for path in item["images"]]
        satellite_image_name = os.path.basename(image_paths[0]) if image_paths else ""
        assert len(image_paths) == 11, f"{region_id} expects 11 images, got {len(image_paths)}"
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{region_id}: image not found -> {path}")

        if model is None or processor is None:
            model, processor = build_model()
            print(torch.cuda.device_count())
            print(getattr(model, "hf_device_map", None))


        try:
            record = infer_one(
                model,
                processor,
                image_paths,
                region_id,
                sample_index=idx,
                satellite_image_name=satellite_image_name,
                source_region_id=source_region_id,
                max_new_tokens=max_new_tokens,
            )
        except Exception:
            record = empty_record(
                region_id,
                sample_index=idx,
                satellite_image_name=satellite_image_name,
                source_region_id=source_region_id,
            )

        if region_id not in seen:
            ordered_records.append(record)
            seen.add(region_id)
        else:
            for pos, row in enumerate(ordered_records):
                if str(row.get("region_id")) == region_id:
                    ordered_records[pos] = record
                    break

        finished = idx - start_idx + 1
        if finished % flush_every == 0:
            save_records(out_json, ordered_records)
        if finished % 10 == 0:
            print(f"done {finished}/{end_idx - start_idx}")

    save_records(out_json, ordered_records)
    print(f"saved to {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract a structured urban semantic profile for each region and store all regions in one JSON file."
    )
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--base_img", type=str, default=BASE_IMG)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--flush_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    BASE_IMG = args.base_img

    main(
        json_path=args.json_path,
        out_json=args.out_json,
        start_idx=int(args.start_idx),
        end_idx=args.end_idx,
        max_new_tokens=int(args.max_new_tokens),
        flush_every=int(args.flush_every),
        resume=bool(args.resume),
    )
