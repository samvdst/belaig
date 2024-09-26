from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the processor and model
processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load and prepare the image
image = Image.open("receipt2.png").convert("RGB")
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
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Convert the output to JSON format
output_json = processor.token2json(output)
print(output_json)
