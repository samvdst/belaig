import argparse
import json
from transformers import DonutProcessor, AutoModelForVision2Seq
from PIL import Image
import torch


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process an image with Donut model.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    # Load the processor and model with trust_remote_code=True to avoid warnings
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2", trust_remote_code=True
    )

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load and prepare the image
    image = Image.open(args.image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Prepare decoder input IDs with the task prompt
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    # Generate output
    generated_ids = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        num_beams=5,
        early_stopping=True,
    )

    # Decode and print the output
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,  # Suppress FutureWarning
    )[0]

    # Convert the output to JSON format
    output_json = processor.token2json(output)

    # Pretty-print the JSON output
    print(json.dumps(output_json, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
