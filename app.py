
from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

@app.route('/preserve-face-edit', methods=['POST'])
def preserve_face_edit():
    data = request.get_json()
    image_url = data.get('image_url')
    instruction = data.get('edit_instruction')

    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return jsonify({"error": "Replicate API token not found in environment."}), 500

    os.environ["REPLICATE_API_TOKEN"] = replicate_token

    try:
        output = replicate.run(
            "cjwbw/stable-diffusion-v2-inpainting:db21e45e327e2e0fb85805c014dff7c2fcdd377d37a1f86e8bcd9d37db78f9df",
            input={
                "prompt": instruction,
                "image": image_url,
                "mask": image_url,
            }
        )
        return jsonify({"edited_image_url": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
