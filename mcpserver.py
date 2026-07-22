"""
MCP Server: storybook-mcp
Exposes the tools used by the LangGraph pipeline:
  write_story | evaluate_story | revise_story
  extract_characters
  generate_image_prompt | evaluate_image_prompt | generate_image
  compose_pdf

All LLM-backed content generation/evaluation lives here so it's reusable
outside this one pipeline and testable in isolation. The pipeline itself
only owns orchestration (routing, human-in-the-loop interrupts, retries).

Run: python mcpserver.py
"""
import asyncio
import json
import re
import os
from pathlib import Path
import logging
import random
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from langchain_openai import ChatOpenAI
from diffusers import FluxPipeline
import torch
    

app = Server("storybook-mcp")
LOG_PATH = Path(__file__).resolve().parent / "mcplogs.log"
mcp_logger = logging.getLogger("storybook-mcp")
mcp_logger.setLevel(logging.INFO)
mcp_logger.propagate = False

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = FlushFileHandler(LOG_PATH, mode="a")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
mcp_logger.addHandler(handler)
mcp_logger.info("OUTSIDE")
mcp_logger.info(f"logger initialized with PID={os.getpid()}")
mcp_logger.info(f"LOG_PATH={LOG_PATH}")

# ── LLM ──────────────────────────────────────────────────────────────────── #

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B-AWQ",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.6,
    top_p=0.8,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.0
    },
)


DEFAULT_STYLE = "children's watercolor illustration"


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _parse_story_json(content: str) -> list[str] | None:
    """Best-effort parse of the LLM's story JSON into a flat list of sentences."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict) and "story" in data and "title" in data:
        return data["story"], data["title"]
   
    return None


def _parse_characters_json(content: str) -> dict | None:
    """Best-effort parse of the LLM's character JSON into {name: {"role":..., "visual_tags":...}}."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict) and "characters" in data:
        chars = data["characters"]
    elif isinstance(data, list):
        chars = data
    else:
        return None

    out = {}
    for c in chars:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        out[name] = {
            "role": c.get("role", ""),
            "visual_tags": c.get("visual_tags", ""),
        }
    return out or None


# ---------- tool registry -------------------------------------------------- #

@app.list_tools()
async def list_tools() -> list[Tool]:
    mcp_logger.info("list_tools called")
    return [
        Tool(
            name="write_story",
            description="Write a children's story for a theme as a flat list of sentences, within a sentence-count range. Optionally addresses prior feedback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "min_sentences": {"type": "integer"},
                    "max_sentences": {"type": "integer"},
                    "feedback": {"type": "string", "default": ""},
                    "story": {"type": "array", "items": {"type": "string"}},
                    "issues": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["theme", "min_sentences", "max_sentences"],
            },
        ),
        Tool(
            name="evaluate_story",
            description="Evaluate a flat story (list of sentences) against a sentence-count range. Returns ok or a list of issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "story": {"type": "array", "items": {"type": "string"}},
                    "min_sentences": {"type": "integer"},
                    "max_sentences": {"type": "integer"},

                },
                "required": ["story", "min_sentences", "max_sentences"],
            },
        ),
        Tool(
            name="extract_characters",
            description="Extract main characters from a story with short, consistent visual traits for illustration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "story": {"type": "array", "items": {"type": "string"}},
                    "style": {"type": "string", "default": DEFAULT_STYLE},
                },
                "required": ["story"],
            },
        ),
        Tool(
            name="generate_image_prompt",
            description="Turn page text into an illustration prompt. If existing_prompt and feedback are given, refines instead of generating fresh. Pass character_tags to keep character appearance consistent across pages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "page_text": {"type": "string"},
                    "style": {"type": "string", "default": DEFAULT_STYLE},
                    "existing_prompt": {"type": "string", "default": ""},
                    "feedback": {"type": "string", "default": ""},
                    "character_tags": {"type": "string", "default": ""},
                },
                "required": ["page_text"],
            },
        ),
        Tool(
            name="evaluate_image_prompt",
            description="Critique an illustration prompt against its page text: vivid? child-safe? no text-in-image? on-topic?",
            inputSchema={
                "type": "object",
                "properties": {
                    "page_text": {"type": "string"},
                    "image_prompt": {"type": "string"},
                },
                "required": ["page_text", "image_prompt"],
            },
        ),
        Tool(
            name="generate_image",
            description="Render an illustration from a prompt and save it to disk. Returns the output path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "required": ["prompt", "output_path"],
            },
        ),
        
    ]


@app.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    mcp_logger.info(
        f"PID={os.getpid()} | call_tool: {name} | args={args}"
    )
    dispatch = {
        "write_story": lambda: _write_story(
            args["theme"], args.get("min_sentences"), args.get("max_sentences"), args.get("feedback", ""), args.get("story",""), args.get("issues","")
        ),
        "evaluate_story": lambda: _evaluate_story(
            args["story"], args.get("min_sentences"), args.get("max_sentences")
        ),
        "extract_characters": lambda: _extract_characters(
            args["story"], args.get("style", DEFAULT_STYLE)
        ),
        "generate_image_prompt": lambda: _generate_image_prompt(
            args["page_text"],
            args.get("style", DEFAULT_STYLE),
            args.get("existing_prompt", ""),
            args.get("feedback", ""),
            args.get("character_tags", ""),
        ),
        "evaluate_image_prompt": lambda: _evaluate_image_prompt(
            args["page_text"], args["image_prompt"]
        ),
        "generate_image": lambda: _generate_image(args["prompt"], args["output_path"]),
       
    }
    fn = dispatch.get(name)
    if fn is None:
        result = {"error": f"unknown tool {name}"}
    else:
        try:
            result = await fn() if asyncio.iscoroutinefunction(fn) else fn()
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:
            mcp_logger.exception(f"call_tool crashed on '{name}'")
            result = {"status": "error", "message": f"{type(e).__name__}: {e}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------- implementations: story ------------------------------------------ #

async def _write_story(theme: str,
                       min_sentences: int,
                       max_sentences: int, 
                       feedback: str, 
                       story: list[str] | None = None,
                       issues: list[str] | None = None) -> dict:
        
        prompt = (
            f"You are an expert children's book author. "
            f"Write an engaging, age-appropriate story based on the theme '{theme}'. "
            )

        if issues:
            prompt += (
                "Revise the existing story to resolve the following issues while preserving the story and charcter descriptions unless changes are necessary.\n"
                f"Issues:\n{json.dumps(issues, indent=2)}\n\n"
            )

        if story:
            prompt += (
                "Existing story:\n"
                f"{json.dumps(story, indent=2)}\n\n"
            )
        
        if feedback:
            prompt += (
                f"Incorporate the following feedback into the story and title:\n{feedback}\n\n"
            )
        
        prompt += (
            f"The story must contain between {min_sentences} and {max_sentences} sentences (inclusive). "
            "Use short, simple sentences suitable for young children. "
            "The story should have a clear beginning, middle, and ending. "
            "Create a short, simple title that accurately reflects the story.\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            '{\n'
            '  "story": ["Sentence 1", "Sentence 2"],\n'
            '  "title": "Story title"\n'
            '}\n'
            "Do not include markdown, code fences, or any additional text."
            )
        
        mcp_logger.info('called write story')
        
        mcp_logger.info(f"prompt: {prompt}\n")
    
        try:
            resp = await llm.ainvoke(prompt)
        except Exception as e:
            mcp_logger.exception("ainvoke failed")
            mcp_logger.info(e)
            raise
        mcp_logger.info(f"op: {resp.content}\n")
        content = _strip_think(resp.content)
        
        sentences, title = _parse_story_json(content)
        if sentences is None:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
    
        return {"output": sentences, "output2": title}


def _evaluate_story(story: list, min_sentences: int, max_sentences: int) -> dict:
    issues = []
    n = len(story)
    if n < min_sentences:
        issues.append(f"Story has {n} sentences; needs at least {min_sentences}.")
    if n > max_sentences:
        issues.append(f"Story has {n} sentences; must be at most {max_sentences}.")
    return {"status": "ok" if not issues else "needs_revision", "message": issues}


async def _extract_characters(story: list[str], style: str) -> dict:
    prompt = (
        "You are a character designer for a children's picture book.\n"
        "Read this story and identify the main characters that appear in illustrations "
        "(skip minor background characters mentioned only in passing).\n\n"
        f"Story:\n{json.dumps(story, indent=2)}\n\n"
        f"Illustration style: {style}\n\n"
        "For each character, give a SHORT, CONCRETE visual description (species/appearance, "
        "colors, one or two distinguishing features, clothing/accessories if any) that can be "
        "reused verbatim in every illustration prompt to keep the character looking the same "
        "across pages. Keep visual_tags under 20 words. Do not include personality or backstory "
        "in visual_tags — appearance only.\n\n"
        "Return ONLY valid JSON, no other text:\n"
        '{"characters": [{"name": "...", "role": "...", "visual_tags": "..."}]}\n'
    )
    mcp_logger.info(f"prompt to extract charcters: {prompt}\n")

    resp = await llm.ainvoke(prompt)
    content = _strip_think(resp.content)
    
    characters = _parse_characters_json(content)
    mcp_logger.info(f"LLM output {characters }\n")
    if characters is None:
        mcp_logger.warning("extract_characters: failed to parse JSON, returning empty dict")
        characters = {}
    return {"output": characters}


# ---------- implementations: images ----------------------------------------- #

async def _generate_image_prompt(
    page_text: str, style: str, existing_prompt: str, feedback: str, character_tags: str = ""
) -> dict:
    char_block = f"Characters: {character_tags}. " if character_tags else ""

    base = (
    "You are an expert illustration prompt writer for children's books. "
    "Generate a concise image-generation prompt optimized for the FLUX.1-schnell model from the provided page text. "
    "You are given visual descriptions for the story's characters. "
    "For every character mentioned in the page text, use the corresponding visual description consistently. "
    "Do not invent new characters or change any character's appearance unless explicitly instructed. "
    "Structure the prompt in this exact order: "
    "style, character appearances, main action, scene, setting, lighting, mood, composition, and important visual details.\n "
    f"Style: {style}.\n "
    f"{char_block}\n"
    f"Page text: {page_text}\n"
    )

    if existing_prompt and feedback:
        base += (
        "Refine the existing prompt using the feedback while preserving all character appearances unless the feedback explicitly requests changes.\n"
        f"Existing prompt: {existing_prompt}\n"
        f"Feedback: {feedback}\n"
        )

    base += (
    "Keep the prompt under 100 words. "
    "Write only the image prompt in natural language. "
    "Do not include explanations, headings, or markdown."
    ) 
    mcp_logger.info(f'Prompt used for image generation : {base}\n')
    resp = await llm.ainvoke(base)
    mcp_logger.info(f'llm output : {resp.content}')
    return {"output": _strip_think(resp.content)}
   

async def _evaluate_image_prompt(page_text: str, image_prompt: str) -> dict:
    resp = await llm.ainvoke(
        "you are an art director for a children's picture book. "
        "Evaluate this illustration prompt: vivid? child-safe? no text in image? "
        "matches page text? optimized for the FLUX.1-schnell model?\n\n"
        f"Page text : {page_text}\n"
        f"Prompt    : {image_prompt}\n\n"
        'Reply ONLY with JSON: {"status":"ok"|"needs_revision","message":"..."}'
    )
    try:
        mcp_logger.info(f"[Evaluate image prompt]:\n {resp.content}")
        raw = _strip_think(resp.content).strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        crit = json.loads(raw)
    except Exception:
        crit = {"status": "ok", "message": ""}
    return crit


# Lazily loaded so the MCP server starts fast and only pays the model-load
# cost the first time an image is actually requested.
_flux_pipe = None


def _get_flux_pipe():
    global _flux_pipe
    
    if _flux_pipe is None:
        
        _flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        
        _flux_pipe.vae.enable_slicing()
        _flux_pipe.vae.enable_tiling()

    else:
        mcp_logger.info('reusing old one')
    
    return _flux_pipe
    


def _generate_image(prompt: str, output_path: str) -> dict:
    try:
        import torch
      
        _flux_pipe = _get_flux_pipe()
     
        image = _flux_pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
        ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return {"status": "ok", "output": output_path}
    except Exception as e:
        return {"status": "error", "message": str(e), "output": output_path}


  

# ---------- main ----------------------------------------------------------- #

async def main():
    
    async with stdio_server() as (r, w):
        mcp_logger.info(f"MCP server starting inside main. PID={os.getpid()}")
        await app.run(r, w, app.create_initialization_options())
if __name__ == "__main__":
    asyncio.run(main())
