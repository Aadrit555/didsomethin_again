import numpy as np
import sys
import os
import json
import tempfile

from PIL import Image
import streamlit as st

# Robust NumPy 2.x compatibility monkeypatch
for attr in ['object', 'bool', 'int', 'float', 'str', 'complex']:
    if not hasattr(np, attr):
        setattr(
            np,
            attr,
            getattr(__builtins__, attr)
            if hasattr(__builtins__, attr)
            else getattr(np, f"{attr}_")
            if hasattr(np, f"{attr}_")
            else None,
        )

st.set_page_config(
    page_title="Temporal Video RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Temporal Video RAG")
st.caption("Upload a video, ask a question, and get grounded, frame-aware answers.")


@st.cache_resource
def load_modules():
    """Initialize heavy modules once and cache them."""
    from ingestion.ingestor import VideoIngestor
    from index.vector_store import VectorStore
    from reasoning.vlm_orchestrator import VLMOrchestrator

    ingestor = VideoIngestor()
    vector_store = VectorStore()
    vlm = VLMOrchestrator()
    return ingestor, vector_store, vlm


# Initialize global state
if "modules_ready" not in st.session_state:
    st.session_state.modules_ready = False
if "processing_state" not in st.session_state:
    st.session_state.processing_state = "idle"  # idle | uploading | processing | done
if "upload_progress" not in st.session_state:
    st.session_state.upload_progress = 0
if "kg_path" not in st.session_state:
    st.session_state.kg_path = None
if "current_video_name" not in st.session_state:
    st.session_state.current_video_name = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

with st.spinner("Initializing AI modules (first run may take a minute)..."):
    ingestor, vector_store, vlm = load_modules()
    st.session_state.modules_ready = True


def render_upload_and_results():
    """Main two-column layout: upload (left) and results (right)."""
    col_upload, col_results = st.columns([1, 2], gap="large")

    # ---------- Upload Panel ----------
    with col_upload:
        st.subheader("1. Upload video", anchor=False)

        st.markdown(
            "Drag and drop a video file or use the **Browse video** button below.  \n"
            "_Supported: MP4, AVI, MOV · up to ~1 GB._"
        )

        uploaded_video = st.file_uploader(
            "Video file",
            type=["mp4", "avi", "mov"],
            help="Upload a local video file to analyze. Larger files may take longer to process.",
            label_visibility="collapsed",
        )

        if uploaded_video is not None:
            st.info(f"Selected file: `{uploaded_video.name}`", icon="📹")

        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        if uploaded_video is not None:
            if (
                st.button(
                    "Process video",
                    type="primary",
                    help="Extract keyframes, build embeddings and the temporal knowledge graph.",
                )
            ):
                st.session_state.processing_state = "processing"
                st.session_state.upload_progress = 0

                with st.spinner("Processing video... (frame extraction & CLIP embeddings)"):
                    # NOTE: Streamlit does not expose true upload progress;
                    # this progress bar reflects back-end processing only.
                    progress_bar = progress_placeholder.progress(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_video.read())
                        video_path = tmp.name

                    progress_bar.progress(15)

                    processed_data = ingestor.process_video(
                        video_path,
                        kg_path=os.path.join("index", f"{uploaded_video.name}_kg.json"),
                    )
                    progress_bar.progress(75)

                    vector_store.add_frames(uploaded_video.name, processed_data["frames"])
                    st.session_state.kg_path = processed_data["kg_path"]
                    st.session_state.current_video_name = uploaded_video.name

                    progress_bar.progress(100)

                st.session_state.processing_state = "done"
                status_placeholder.success(
                    f"Processed {len(processed_data['frames'])} frames and built a knowledge graph for `{uploaded_video.name}`."
                )
        else:
            status_placeholder.info(
                "Waiting for a video. Upload a file to begin.", icon="⏳"
            )

        # Lightweight status indicator
        state = st.session_state.processing_state
        state_label = {
            "idle": "Idle · no video processed yet",
            "processing": "Processing video…",
            "done": "Ready · video processed",
            "uploading": "Uploading video…",
        }.get(state, "Idle")

        st.caption(f"Status: **{state_label}**")

    # ---------- Results Panel ----------
    with col_results:
        st.subheader("3. Results", anchor=False)

        container = st.container(border=True)
        with container:
            if st.session_state.last_answer is None:
                st.write(
                    "Once you upload a video and ask a question, the grounded answer and evidence will appear here."
                )
            else:
                st.markdown("**Latest question**")
                st.write(st.session_state.last_query)

                st.markdown("**Answer**")
                st.write(st.session_state.last_answer)

                # Simple JSON download for programmatic use
                payload = {
                    "query": st.session_state.last_query,
                    "answer": st.session_state.last_answer,
                    "video": st.session_state.current_video_name,
                }
                st.download_button(
                    "Download answer as JSON",
                    data=json.dumps(payload, indent=2),
                    file_name="video_answer.json",
                    mime="application/json",
                    type="secondary",
                )


def render_query_panel():
    """Bottom query area where the user asks about the processed video."""
    st.markdown("---")
    st.subheader("2. Ask a question", anchor=False)

    query = st.text_input(
        "What would you like to know about this video?",
        placeholder=(
            "Examples: 'Summarize the video', "
            "'What happens in the second half?', "
            "'How does the scene change after 30 seconds?'"
        ),
        label_visibility="collapsed",
        key="query_input",
    )

    col_btn, col_hint = st.columns([1, 3])

    with col_btn:
        analyze_clicked = st.button(
            "Analyze video",
            type="primary",
            disabled=not query.strip(),
        )

    with col_hint:
        st.caption(
            "Tip: Ask about specific moments (e.g., *'What is different between the first and second half of the video?'*)."
        )

    if analyze_clicked and query.strip():
        if not st.session_state.kg_path:
            st.warning("Please upload and process a video before asking questions.")
            return

        with st.spinner("Searching, retrieving frames, and reasoning over the video..."):
            # 1. Embed query
            query_embedding = ingestor.model.encode(query).tolist()

            # 2. Vector search
            search_results = vector_store.search(query_embedding, n_results=3)

            if search_results["metadatas"][0]:
                top_meta = search_results["metadatas"][0][0]
                timestamp = top_meta["timestamp"]
                video_id = top_meta["video_id"]

                # 3. Temporal context
                context_frames = vector_store.get_temporal_context(
                    timestamp, video_id, window_seconds=2
                )

                # 4. VLM reasoning (Dual-Channel)
                answer = vlm.generate_answer(
                    query,
                    context_frames,
                    knowledge_graph_path=st.session_state.get("kg_path"),
                )

                st.session_state.last_query = query
                st.session_state.last_answer = answer

                st.success(
                    f"Found relevant evidence around {timestamp:.2f}s in `{video_id}`."
                )

                # Show answer inline so users don't have to look at the Results panel
                if isinstance(answer, str) and answer.strip():
                    if answer.strip().startswith("❌ Error") or answer.strip().startswith(
                        "Error"
                    ):
                        st.error(answer)
                    else:
                        st.markdown("#### Answer")
                        st.write(answer)
                else:
                    st.warning(
                        "The model did not return a readable answer. "
                        "Please verify that your VLM (e.g. Ollama with `llava`) is running."
                    )

                # Evidence preview
                if context_frames:
                    st.markdown("#### Evidence frames")
                    cols = st.columns(len(context_frames))
                    for i, frame in enumerate(context_frames):
                        img = Image.open(frame["frame_path"])
                        cols[i].image(
                            img,
                            caption=f"t = {frame['timestamp']:.2f}s",
                        )
            else:
                st.warning("No matches found in the video database for this query.")


def render_animation_playground():
    """Interactive animation playground demonstrating 2D, 3D and UI animations."""
    st.subheader("Animation playground", anchor=False)
    st.caption(
        "This demo shows 2D character motion, environmental effects, a simple 3D scene, "
        "and UI micro‑interactions, with accessibility and performance considerations."
    )

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("**Controls**")
        reduce_motion = st.checkbox(
            "Reduce motion (accessibility)",
            help="Disables large camera moves and heavy particle effects.",
            value=False,
        )
        enable_3d = st.checkbox(
            "Enable 3D cube demo",
            help="Turn off if your device struggles with WebGL.",
            value=True,
        )
        particle_density = st.slider(
            "Particle density",
            min_value=0,
            max_value=200,
            value=80,
            step=10,
            help="Controls how many particles are emitted in the 2D scene.",
        )

    settings = {
        "reduceMotion": reduce_motion,
        "enable3D": enable_3d,
        "particleDensity": particle_density,
    }

    with col_left:
        html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      :root {{
        color-scheme: dark;
        --bg: #020617;
        --panel: #020617;
        --border: #1e293b;
        --accent: #6366f1;
        --accent-soft: rgba(99, 102, 241, 0.18);
        --text: #e5e7eb;
        --muted: #9ca3af;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
          "Segoe UI", sans-serif;
        background: transparent;
        color: var(--text);
      }}

      .playground-root {{
        display: grid;
        grid-template-columns: minmax(0, 2fr) minmax(0, 1.4fr);
        gap: 16px;
        padding: 12px;
        border-radius: 16px;
        background: radial-gradient(circle at top left, #0f172a 0, #020617 50%);
        border: 1px solid var(--border);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.65);
        min-height: 320px;
      }}

      .scene2d-wrapper {{
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(15, 23, 42, 0.9);
        background: linear-gradient(180deg, #020617 0%, #020617 45%, #0b1120 100%);
      }}

      #scene2d {{
        width: 100%;
        height: 260px;
        display: block;
      }}

      .scene2d-overlay {{
        position: absolute;
        inset: 0;
        pointer-events: none;
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding: 10px 12px;
        font-size: 11px;
        color: var(--muted);
        background: linear-gradient(180deg, transparent 0%, rgba(0, 0, 0, 0.6) 100%);
      }}

      .chip-row {{
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
      }}

      .chip {{
        padding: 3px 8px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(8px);
      }}

      .ui-panel {{
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}

      .ui-card {{
        border-radius: 12px;
        border: 1px solid rgba(30, 64, 175, 0.6);
        background: radial-gradient(circle at top, #0b1120 0, #020617 60%);
        padding: 10px 10px 12px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
      }}

      .ui-card h3 {{
        margin: 0 0 4px;
        font-size: 12px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      .ui-card p {{
        margin: 0 0 8px;
        font-size: 11px;
        color: var(--muted);
      }}

      .button-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}

      .btn-animated {{
        position: relative;
        padding: 6px 14px;
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #4f46e5, #22d3ee);
        color: white;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        outline: none;
        box-shadow: 0 10px 28px rgba(56, 189, 248, 0.45);
        transform-origin: center;
        transition:
          transform 160ms ease-out,
          box-shadow 160ms ease-out,
          filter 160ms ease-out;
      }}

      .btn-animated:hover {{
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 16px 45px rgba(56, 189, 248, 0.6);
        filter: brightness(1.05);
      }}

      .btn-animated:active {{
        transform: translateY(0) scale(0.98);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.9);
        filter: brightness(0.97);
      }}

      .pill {{
        font-size: 10px;
        padding: 3px 7px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: var(--muted);
      }}

      .status-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-top: 4px;
        font-size: 10px;
        color: var(--muted);
      }}

      .status-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 999px;
        margin-right: 4px;
        background: #22c55e;
      }}

      .toggle {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 10px;
        color: var(--muted);
        cursor: pointer;
      }}

      .toggle input {{
        accent-color: var(--accent);
      }}

      .scene3d {{
        margin-top: 10px;
        border-radius: 10px;
        border: 1px solid rgba(30, 64, 175, 0.6);
        overflow: hidden;
        position: relative;
      }}

      .scene3d canvas {{
        display: block;
        width: 100%;
        height: 150px;
      }}

      .scene3d-label {{
        position: absolute;
        left: 8px;
        bottom: 6px;
        font-size: 10px;
        color: var(--muted);
        background: rgba(15, 23, 42, 0.75);
        padding: 2px 6px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.3);
      }}

      @media (max-width: 700px) {{
        .playground-root {{
          grid-template-columns: minmax(0, 1fr);
        }}

        #scene2d {{
          height: 220px;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="playground-root">
      <div class="scene2d-wrapper">
        <canvas id="scene2d"></canvas>
        <div class="scene2d-overlay">
          <div class="chip-row">
            <span class="chip">Keyframed motion</span>
            <span class="chip">Tweening & easing</span>
            <span class="chip">Particles</span>
          </div>
          <span style="font-size:10px;">2D · character & weather loop</span>
        </div>
      </div>
      <div class="ui-panel">
        <div class="ui-card">
          <h3>UI micro‑interactions</h3>
          <p>
            Hover, press, and system feedback support character and environmental motion
            without overwhelming the user.
          </p>
          <div class="button-row">
            <button class="btn-animated" id="btnTrigger">
              Trigger on‑click pulse
            </button>
            <span class="pill">Hover &amp; click animations</span>
          </div>
          <div class="status-row">
            <span><span class="status-dot"></span>Timeline running</span>
            <label class="toggle">
              <input id="toggleAnimations" type="checkbox" />
              Pause all motion
            </label>
          </div>
        </div>

        <div class="scene3d" id="scene3dContainer">
          <div class="scene3d-label">3D · lit cube with smooth rotation</div>
        </div>
      </div>
    </div>

    <!-- Three.js for simple 3D demo -->
    <script src="https://unpkg.com/three@0.161.0/build/three.min.js"></script>
    <script>
      const settings = {json.dumps(settings)};

      const prefersReduced =
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const effectiveReduceMotion = settings.reduceMotion || prefersReduced;

      // --------------------- 2D ANIMATION ENGINE ---------------------
      const canvas = document.getElementById("scene2d");
      const ctx = canvas.getContext("2d");

      function resizeCanvas() {{
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }}

      resizeCanvas();
      window.addEventListener("resize", resizeCanvas);

      const character = {{
        x: 0,
        y: 0,
        baseY: 0,
        speed: 60,
        t: 0,
      }};

      const particles = [];
      const maxParticles = settings.particleDensity;

      const timeline = {{
        time: 0,
        duration: 6,
        playing: true,
      }};

      const Easing = {{
        easeInOutQuad: (t) =>
          t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
      }};

      function emitParticle(width, height) {{
        if (particles.length > maxParticles) return;
        const x = Math.random() * width;
        const y = -10 - Math.random() * 40;
        particles.push({{
          x,
          y,
          vy: 120 + Math.random() * 60,
          life: 0,
          maxLife: 2 + Math.random() * 1.5,
        }});
      }}

      function updateParticles(dt, width, height) {{
        for (let i = particles.length - 1; i >= 0; i--) {{
          const p = particles[i];
          p.y += p.vy * dt;
          p.life += dt;
          if (p.y > height + 20 || p.life > p.maxLife) {{
            particles.splice(i, 1);
          }}
        }}
        if (!effectiveReduceMotion) {{
          emitParticle(width, height);
        }}
      }}

      function drawScene2D(dt) {{
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        timeline.time = (timeline.time + dt) % timeline.duration;
        const tNorm = timeline.time / timeline.duration;

        const walkPhase = (tNorm * 2 * Math.PI) % (2 * Math.PI);
        const hop = Math.sin(walkPhase) * 6;

        const eased = Easing.easeInOutQuad(tNorm);
        const pathMargin = 40;
        const pathWidth = width - pathMargin * 2;
        const x = pathMargin + eased * pathWidth;

        character.x = x;
        character.baseY = height * 0.6;
        character.y = character.baseY + hop;

        updateParticles(dt, width, height);

        // Background: sky gradient with a simple day-night tint
        const g = ctx.createLinearGradient(0, 0, 0, height);
        const nightFactor = Math.abs(Math.sin(tNorm * Math.PI));
        const topColor =
          nightFactor > 0.5 ? "#020617" : "#0f172a";
        const bottomColor =
          nightFactor > 0.5 ? "#020617" : "#020617";
        g.addColorStop(0, topColor);
        g.addColorStop(1, bottomColor);
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, width, height);

        // Ground
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, height * 0.62, width, height * 0.4);

        // Particles (rain)
        ctx.strokeStyle = "rgba(148, 163, 184, 0.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const p of particles) {{
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p.x + 2, p.y + 7);
        }}
        ctx.stroke();

        // Character: simple rounded figure with a "walk" bob
        ctx.save();
        ctx.translate(character.x, character.y);

        ctx.shadowColor = "rgba(15,23,42,0.75)";
        ctx.shadowBlur = 16;
        ctx.shadowOffsetY = 6;

        ctx.beginPath();
        ctx.fillStyle = "#4f46e5";
        ctx.arc(0, -18, 10, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(-8, -10);
        ctx.lineTo(-8, 8);
        ctx.lineTo(8, 8);
        ctx.lineTo(8, -10);
        ctx.closePath();
        const bodyGrad = ctx.createLinearGradient(-8, -10, 8, 8);
        bodyGrad.addColorStop(0, "#6366f1");
        bodyGrad.addColorStop(1, "#22d3ee");
        ctx.fillStyle = bodyGrad;
        ctx.fill();

        // Legs (simple swinging motion)
        const legOffset = Math.sin(walkPhase) * 4;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#e5e7eb";
        ctx.beginPath();
        ctx.moveTo(-4, 8);
        ctx.lineTo(-4 - legOffset, 16);
        ctx.moveTo(4, 8);
        ctx.lineTo(4 + legOffset, 16);
        ctx.stroke();

        ctx.restore();

        // Simple shadow under character
        ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
        ctx.beginPath();
        ctx.ellipse(character.x, character.baseY + 16, 14, 5, 0, 0, Math.PI * 2);
        ctx.fill();
      }}

      let lastTime = performance.now();
      function loop(now) {{
        const dt = Math.min((now - lastTime) / 1000, 0.033);
        lastTime = now;
        if (timeline.playing) {{
          drawScene2D(dt);
        }}
        requestAnimationFrame(loop);
      }}
      requestAnimationFrame(loop);

      // --------------------- UI INTERACTION ---------------------------
      const toggle = document.getElementById("toggleAnimations");
      const btnTrigger = document.getElementById("btnTrigger");

      toggle.checked = false;
      toggle.addEventListener("change", () => {{
        timeline.playing = !toggle.checked;
      }});

      btnTrigger.addEventListener("click", () => {{
        if (effectiveReduceMotion) return;
        btnTrigger.style.transition = "transform 260ms cubic-bezier(0.34, 1.56, 0.64, 1)";
        btnTrigger.style.transform = "scale(1.08) translateY(-1px)";
        setTimeout(() => {{
          btnTrigger.style.transform = "";
        }}, 260);
      }});

      // --------------------- 3D SCENE (Three.js) ---------------------
      if (settings.enable3D && window.THREE) {{
        const container = document.getElementById("scene3dContainer");
        const width = container.clientWidth;
        const height = 150;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x020617);

        const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
        camera.position.set(2.2, 1.6, 3.2);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
        renderer.shadowMap.enabled = !effectiveReduceMotion;
        container.appendChild(renderer.domElement);

        const light = new THREE.DirectionalLight(0xffffff, 1.1);
        light.position.set(2, 4, 3);
        light.castShadow = !effectiveReduceMotion;
        scene.add(light);

        const ambient = new THREE.AmbientLight(0x64748b, 0.7);
        scene.add(ambient);

        const planeGeo = new THREE.PlaneGeometry(8, 4);
        const planeMat = new THREE.MeshStandardMaterial({{
          color: 0x020617,
          roughness: 0.9,
          metalness: 0.1,
        }});
        const plane = new THREE.Mesh(planeGeo, planeMat);
        plane.receiveShadow = !effectiveReduceMotion;
        plane.rotation.x = -Math.PI / 2;
        plane.position.y = -1;
        scene.add(plane);

        const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
        const cubeMat = new THREE.MeshStandardMaterial({{
          color: 0x6366f1,
          metalness: 0.4,
          roughness: 0.25,
        }});
        const cube = new THREE.Mesh(cubeGeo, cubeMat);
        cube.castShadow = !effectiveReduceMotion;
        cube.position.y = -0.3;
        scene.add(cube);

        function render3D(now) {{
          const t = now * 0.001;
          if (!effectiveReduceMotion) {{
            cube.rotation.y = t * 0.7;
            cube.rotation.x = Math.sin(t * 0.6) * 0.4;
          }}
          renderer.render(scene, camera);
          requestAnimationFrame(render3D);
        }}
        requestAnimationFrame(render3D);
      }} else {{
        const container = document.getElementById("scene3dContainer");
        container.innerHTML =
          '<div style="padding:8px;font-size:11px;color:var(--muted);">3D demo disabled for this device.</div>';
      }}
    </script>
  </body>
</html>
        """
        components.html(html, height=340)


def main():
    render_upload_and_results()
    render_query_panel()


if __name__ == "__main__":
    main()
