import torch
from diffusers import FluxPipeline
from PIL import Image
import os
from flask import Flask, request, render_template, send_from_directory, jsonify, session, make_response
import uuid
import glob
import traceback
import logging
import time
import shutil
from datetime import timedelta

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Secret key and session configuration
app.secret_key = os.environ.get('FLASK_SECRET_KEY', str(uuid.uuid4()))
app.permanent_session_lifetime = timedelta(hours=1)

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Directories
STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Pipeline global variable
pipe = None

# -----------------------------
# Helpers
# -----------------------------

def _get_device_from_obj(obj):
    """Return device if obj or any nested item has a .device attribute."""
    try:
        if obj is None:
            return None
        if hasattr(obj, "device"):
            return obj.device
        if isinstance(obj, (tuple, list)):
            for item in obj:
                d = _get_device_from_obj(item)
                if d is not None:
                    return d
        if isinstance(obj, dict):
            for v in obj.values():
                d = _get_device_from_obj(v)
                if d is not None:
                    return d
    except Exception as e:
        logger.warning(f"Error in _get_device_from_obj: {str(e)}")
        return None
    return None

def get_session_dir():
    """Create a private static directory for each user session."""
    if "session_id" not in session:
        session.permanent = True
        session["session_id"] = uuid.uuid4().hex
    session_dir = os.path.join(STATIC_DIR, session["session_id"])
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def cleanup_old_sessions():
    """Remove session directories older than the session lifetime."""
    try:
        now = time.time()
        for session_dir in glob.glob(os.path.join(STATIC_DIR, "*")):
            if os.path.isdir(session_dir):
                mtime = os.path.getmtime(session_dir)
                if now - mtime > app.permanent_session_lifetime.total_seconds():
                    shutil.rmtree(session_dir, ignore_errors=True)
                    logger.info(f"Cleaned up old session directory: {session_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up old sessions: {str(e)}")

# -----------------------------
# Load pipeline once
# -----------------------------
def load_pipeline():
    global pipe
    try:
        logger.info("Loading FluxPipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )

        # Memory optimizations
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.vae.enable_tiling()

        # Debug hook
        def debug_device(module, input, output):
            dev = _get_device_from_obj(input) or _get_device_from_obj(output)
            try:
                if dev is None:
                    params = list(module.parameters())
                    if params:
                        dev = params[0].device
            except Exception:
                pass
            logger.debug(f"Active module ({module.__class__.__name__}) device: {dev}")

        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            try:
                pipe.transformer.register_forward_hook(debug_device)
            except Exception as hook_e:
                logger.warning(f"Failed to register forward hook: {hook_e}")

        try:
            first_param = next(pipe.transformer.parameters())
            logger.info(f"Initial model device: {first_param.device}")
        except StopIteration:
            logger.warning("Transformer has no parameters.")
        except Exception as e:
            logger.error(f"Error checking initial model device: {e}")

    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        raise

try:
    load_pipeline()
except Exception as e:
    logger.critical(f"Failed to initialize application: {str(e)}")
    raise

# -----------------------------
# Routes
# -----------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    error = None
    prompt = request.form.get('prompt', '').strip() if request.method == 'POST' else ""

    # Clean up old sessions
    cleanup_old_sessions()

    session_dir = get_session_dir()
    gallery_images = [
        f"/static/{session['session_id']}/{os.path.basename(f)}"
        for f in sorted(glob.glob(os.path.join(session_dir, "*.png")), key=os.path.getmtime, reverse=True)
    ]

    if request.method == 'POST':
        logger.info(f"Processing prompt: {prompt[:50]}...")
        if not prompt:
            error = "Prompt cannot be empty."
        elif len(prompt) > 512:
            error = "Prompt is too long (max 512 characters)."
        else:
            try:
                forbidden_words = ['inappropriate', 'offensive']  # Expand as needed
                if any(word in prompt.lower() for word in forbidden_words):
                    error = "Prompt contains inappropriate content."
                else:
                    generator = torch.Generator(device="cpu").manual_seed(int(time.time()))

                    with torch.no_grad():
                        result = pipe(
                            prompt,
                            guidance_scale=1.0,
                            num_inference_steps=10,
                            max_sequence_length=512,
                            width=800,
                            height=800,
                            generator=generator
                        )

                    if not getattr(result, "images", None) or len(result.images) == 0:
                        raise ValueError("No images generated by pipeline")

                    image = result.images[0]
                    image_filename = f"generated_{uuid.uuid4().hex}.png"
                    image_path = os.path.join(session_dir, image_filename)
                    image.save(image_path, quality=95)

                    image_url = f"/static/{session['session_id']}/{image_filename}"
                    gallery_images.insert(0, image_url)

                    # Limit images per session
                    max_images = 50
                    if len(gallery_images) > max_images:
                        old_images = sorted(
                            glob.glob(os.path.join(session_dir, "*.png")),
                            key=os.path.getmtime
                        )
                        for old_image in old_images[:-max_images]:
                            try:
                                os.remove(old_image)
                            except Exception as e:
                                logger.warning(f"Failed to remove old image {old_image}: {e}")

            except Exception as e:
                error = f"Error generating image: {str(e)}"
                logger.error(f"Detailed error: {traceback.format_exc()}")

    logger.info("Rendering index.html with single textarea")
    response = make_response(render_template(
        'index.html',
        image_url=image_url,
        prompt=prompt,
        error=error,
        gallery_images=gallery_images[:50]
    ))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

@app.route('/static/<session_id>/<filename>')
def serve_image(session_id, filename):
    if "session_id" not in session or session["session_id"] != session_id:
        logger.warning(f"Unauthorized access attempt to {session_id}/{filename}")
        return "Unauthorized", 403
    file_path = os.path.join(STATIC_DIR, session_id, filename)
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return "File not found", 404
    return send_from_directory(os.path.join(STATIC_DIR, session_id), filename)

@app.route('/gallery')
def gallery():
    session_dir = get_session_dir()
    gallery_images = [
        f"/static/{session['session_id']}/{os.path.basename(f)}"
        for f in sorted(glob.glob(os.path.join(session_dir, "*.png")), key=os.path.getmtime, reverse=True)
    ]
    return jsonify({"images": gallery_images[:50]})

@app.route('/clear_gallery', methods=['POST'])
def clear_gallery():
    """Clear all images from current session."""
    session_dir = get_session_dir()
    for img_file in glob.glob(os.path.join(session_dir, "*.png")):
        try:
            os.remove(img_file)
        except Exception as e:
            logger.warning(f"Failed to delete {img_file}: {e}")
    return jsonify({"success": True})

@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"404 error: {str(error)}")
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return render_template('error.html', message="Internal server error"), 500

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)