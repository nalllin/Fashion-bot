import os
import replicate

print("REPLICATE_API_TOKEN set?", bool(os.environ.get("REPLICATE_API_TOKEN")))

MODEL_ID = "bytedance/sdxl-lightning-4step:6f7a773af6fc3e8de9d5a3c00be77c17308914bf67772726aff83496ba1e3bbe"

prompt = (
    "high quality fashion photo of a soft girl college outfit, "
    "white sneakers, pastel colors, full body, pinterest aesthetic"
)

print("Calling Replicate SDXL-Lightning...")
output = replicate.run(
    MODEL_ID,
    input={
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "num_outputs": 1,
        "guidance_scale": 0,
        "num_inference_steps": 4,
        "scheduler": "K_EULER_ANCESTRAL",
        "negative_prompt": "low quality, blurry, distorted, ugly",
    },
)

print("Raw output:", output)

if isinstance(output, list) and output:
    print("\n✅ SDXL image URL:", output[0])
else:
    print("\n⚠️ Unexpected SDXL output shape")
