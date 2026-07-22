"""
Storybook Pipeline — LangGraph + MCP + Human-in-the-Loop
"""
# from huggingface_hub import login

# login("HF_KEY") #FLUX .1 Schnell needs hf login token.


from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import logging
from typing import Any, Literal, TypedDict
from IPython.display import display, Image as IPImage
from langgraph.types import interrupt
from langgraph.graph import END, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
from contextlib import AsyncExitStack
from langchain_mcp_adapters.tools import load_mcp_tools

import subprocess
import signal
import time
import re

logging.basicConfig(
    filename="pipelinelogs.log",
    format="%(asctime)s %(levelname)s: %(message)s",
    filemode="w",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


#----------CONSTANTS----------

MAX_STORY_REVISIONS = 2
MAX_IMG_REVISIONS = 2


#------------------------------
#-------------------Vllm-----------------
#------------------------------


class VLLMServer:
    def __init__(self):
        self.process = None

    def start(self):
        
        model = "Qwen/Qwen3-8B-AWQ"
        logger.info(f'vllm server is called {model}')
        env = os.environ.copy()

        env["HF_HUB_DISABLE_XET"] = "1"
        
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = "1"   # explicit, using 2nd GPU device
        
        
        log_file = open("vllm.log", "w")
        self.process = subprocess.Popen(
            [
                "vllm", "serve",
                model,
                "--port", "8000",
                "--gpu-memory-utilization", "0.85",
                "--max-model-len", "16384",
                "--enforce-eager",
                "--attention-backend", "TRITON_ATTN",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )
        print("vLLM started with PID:", self.process.pid)

    def stop(self):
        logger.info('vllm server is killed.')
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

#------------------------------------------------
#--------------MCP protocol and Vllm server  calling
#--------------------------

@dataclass
class PipelineContext:
    mcp_client: MultiServerMCPClient
    tools: dict[str, Any]
    server: VLLMServer
    mcp_config: dict
    _exit_stack: AsyncExitStack   # keeps the session alive

    @classmethod
    async def create(cls, mcp_config: dict) -> "PipelineContext":
        server = VLLMServer()
        server.start()
        time.sleep(3 * 60)

        mcp_client = MultiServerMCPClient(mcp_config)

        # Open ONE persistent session and keep it open for the run's lifetime.
        exit_stack = AsyncExitStack()
        session = await exit_stack.enter_async_context(
            mcp_client.session("storybook")
        )
        raw_tools = await load_mcp_tools(session)
        tools = {t.name: t for t in raw_tools}

        logger.info(f"[MCP] Connected — tools: {list(tools)}")
        return cls(
            mcp_client=mcp_client,
            tools=tools,
            server=server,
            mcp_config=mcp_config,
            _exit_stack=exit_stack,
        )

    def stop_server(self):
        logger.info("[PipelineContext] stopping vLLM server...")
        if self.server is not None:
            self.server.stop()

    def restart_server(self, wait_seconds: int = 2 * 60):
        logger.info("[PipelineContext] Restarting vLLM server...")
        self.server = VLLMServer()
        self.server.start()
        time.sleep(wait_seconds)

    async def shutdown(self):
        self.stop_server()
        await self._exit_stack.aclose()   # cleanly closes the MCP subprocess/session

# ══════════════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════════════

PageVerdict = Literal["ok", "image_only"]
VALID_VERDICTS = {"ok", "image_only"}

class PageState(TypedDict):
    page_number: int
    text: str
    image_status : str
    image_prompt: str
    image_path: str
    image_feedback: str
    image_revisions: int
    verdict: PageVerdict
    human_note: str


class CharacterTraits(TypedDict):
    role: str
    visual_tags: str


class StorybookState(TypedDict):
    # ── Inputs ──
    theme: str
    num_pages: int
    title: str
    min_sentences: int
    max_sentences: int

    # ── Raw story (flat list of sentences, pre-pagination) ──
    story: list[str]
    story_evaluation: dict
    story_revisions: int

    # ── Human story-level review ──
    human_story_feedback: str  # "ok" or free-text → triggers rewrite
    story_rewrite_reason: str

    # ── Characters (populated once, after story is human-approved) ──
    characters: dict[str, CharacterTraits]  # {name: {"role": ..., "visual_tags": ...}}

    # ── Pages (post-split) ──
    pages: list[PageState]

    # ── Routing (set by human_review_pages) ──
    pages_needing_image: list[int]

    # ── Output ──
    pdf_path: str
    errors: list[str]


# ══════════════════════════════════════════════════════════════════════════════
#  MCP HELPER
# ══════════════════════════════════════════════════════════════════════════════

async def _call_tool(tools: dict, name: str, **kwargs) -> Any:
    tool = tools.get(name)
    if not tool:
        raise ValueError(f"MCP tool '{name}' not registered.")
    result = await tool.ainvoke(kwargs)
    return json.loads(result) if isinstance(result, str) else result

def unwrapped_mcpcontent(mcp_output:list[dict]) -> str:
    block = mcp_output[0]
    if block.get("type") == "text":
        op = json.loads(block["text"])
    else:
        logger.warning("No text block")
        op = []
    return op.get("output",''), op.get("output2",''), op.get('status',''), op.get('message','')


def _character_tags_str(characters: dict) -> str:
    """Flatten the characters dict into a single string safe to inject into
    an image prompt. Always reuses the exact same visual_tags phrasing per
    character so repeated pages describe the character identically."""
    if not characters:
        return ""
    parts = [
        f"{name} (visual description: {c['visual_tags']})"
        for name, c in characters.items()
        if c.get("visual_tags")
    ]
    return "; ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  NODES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Story generation ────────────────────────────────────────────────────── #
async def story_node(state: StorybookState, tools: dict) -> StorybookState:
    logger.info(f"[Story] {'Rewriting' if state.get("story_rewrite_reason", "") else 'Generating'} — theme: '{state['theme']}'")
    print(f"[Story] {'Rewriting' if state.get("story_rewrite_reason", "") else 'Generating'} — theme: '{state['theme']}'")

    result = await _call_tool(
        tools,
        "write_story",
        theme=state["theme"],
        min_sentences=state["min_sentences"],
        max_sentences=state["max_sentences"] ,
        feedback= state.get("story_rewrite_reason", "")
    )

    sentences,title, _ , _ = unwrapped_mcpcontent(result)
    
    logger.info(f"[Story] {len(sentences)} sentences generated")
    logger.info(f"[Story] Title : {title}")



    return {
        **state,
        "story": sentences,
        "title": title or f"The Story of {state['theme'].title()}",
        "story_revisions": state.get("story_revisions", 0),
        "story_rewrite_reason": "",
    }


# ── 2. Reflect-story  ──────────────────────────────────── #
async def reflect_story_node(state: StorybookState, tools: dict) -> StorybookState:
    logger.info("[Reflect-Story] Evaluating story...")
    print("[Reflect-Story] Evaluating story...")
    
    ev = await _call_tool(
        tools,
        "evaluate_story",
        story=state["story"],
        min_sentences=state["min_sentences"],
        max_sentences=state["max_sentences"],
    )
    # mcp returns back as a text wrapper
    _,_, status, message = unwrapped_mcpcontent(ev)
    ev = {'status': status, 'message' : message}
    if status == "needs_revision" and state["story_revisions"] < MAX_STORY_REVISIONS:
        result = await _call_tool(
            tools,
            "write_story",
            theme=state["theme"],
            min_sentences=state["min_sentences"],
            max_sentences=state["max_sentences"],
            feedback = state.get("story_rewrite_reason", ""),
            story= state["story"],
            issues= message
        )
        sentences,_, _ , _ = unwrapped_mcpcontent(result)
        return {
            **state,
            "story": sentences,
            "story_evaluation": ev,
            "story_revisions": state["story_revisions"] + 1,
        }


    return {**state, "story_evaluation": ev}


# ── 3. Human review — story (pre-split) ────────────────────────────────────── #
async def human_review_story_node(state: StorybookState) -> StorybookState:
    logger.info('Human reviewing node')
    print('Human reviewing node') 
    feedback: str = interrupt(
        {
            "prompt": "Type 'ok' to proceed, or describe what to change:",
            "story": state["story"],
        }
    )
    return {**state, "human_story_feedback": str(feedback).strip()}


# ── 4. Extract characters (runs once story is human-approved) ──────────────── #
async def extract_characters_node(state: StorybookState, tools: dict) -> StorybookState:
    logger.info("[Characters] Extracting character traits...")
    print("[Characters] Extracting character traits...")
    
    result = await _call_tool(tools, "extract_characters", story=state["story"])
    characters,_, status, message = unwrapped_mcpcontent(result)

    if not isinstance(characters, dict) or not characters:
        logger.warning(f"[Characters] extraction returned nothing usable: {message}")
        characters = {}

    logger.info(f"[Characters] found: {list(characters.keys())}")
    for name, traits in characters.items():
        logger.info(f"{name}: {traits.get('visual_tags', '')}")

    return {**state, "characters": characters}


# ── 5. Split story sentences into pages ────────────────────────────────────── #
async def split_pages_node(state: StorybookState, tools: dict) -> StorybookState:
    sentences = state["story"]
    num_pages = max(1, state["num_pages"])
    n = len(sentences)
    pages: list[PageState] = []
    logger.info(f"[Split] Distributing {n} sentences across {num_pages} pages")
    print(f"[Split] Distributing {n} sentences across {num_pages} pages")
    
    pages.append(
        PageState(
                page_number=0,
                text = state["theme"],
                image_prompt="",
                image_path="",
                image_feedback="",
                image_revisions=0,
                verdict="ok",
                human_note="",
            )
        ) #for theme page
    
    # Even distribution with remainder spread across the first pages.
    base, remainder = divmod(n, num_pages)
    idx = 0
    for page_num in range(1, num_pages + 1):
        take = base + (1 if page_num <= remainder else 0)
        chunk = sentences[idx: idx + take]
        idx += take
        pages.append(
            PageState(
                page_number=page_num,
                text=" ".join(chunk).strip(),
                image_prompt="",
                image_path="",
                image_feedback="",
                image_revisions=0,
                verdict="ok",
                human_note="",
            )
        )

    return {**state, "pages": pages}


    
# ── 6. Image generation (full run OR per-page rerun) ───────────────────────── #
async def image_prompt_node(
    state: StorybookState, tools: dict, only_pages: list[int] | None = None
) -> StorybookState:
    
    target = set(only_pages) if only_pages else None # only work after human review
    logger.info(f"[IMAGE PROMPT NODE] Target pages are {target}")
    logger.info(f"[IMAGE PROMPT NODE] Generating prompts {'for ' + str(sorted(target)) if target else '(all)'}")

    print(f"[IMAGE PROMPT NODE] Generating prompts {'for ' + str(sorted(target)) if target else '(all)'}")

    char_tags = _character_tags_str(state.get("characters", {}))
    if char_tags:
        logger.info(f"[IMAGE PROMPT NODE] Using character tags: {char_tags}")

    updated: list[PageState] = []
  
    for page in state["pages"]:
        logger.info(f"[IMAGE PROMPT NODE] Page number : {page["page_number"]}")
        
        if target and page["page_number"] not in target:
            updated.append(page)
            continue

        if page.get('image_status') == "ok": # dont generate prompt again
            updated.append(page)
            continue

        if not page.get('image_status'): # for the intial prompt generation  
            note = page.get("human_note", "")
            page_text= page["text"] + (f"(Note: {note})" if note else "")

            image_node_op = await _call_tool(
                tools,
                "generate_image_prompt",
                page_text= page["text"] + (f"(Note: {note})" if note else ""),
                character_tags= char_tags,
            )
            
            image_prompt, _ , _ , _ = unwrapped_mcpcontent(image_node_op)
            
            logger.info(f"[IMAGE PROMPT NODE] GENERATED image prompt of page text: \n {page["text"]} \n --> \n {image_prompt}\n")
        
            updated.append({**page, 
                            "image_prompt": image_prompt,
                            "image_feedback": "",
                            "image_status": ""})

        
        if page.get("image_feedback") and page.get("image_prompt") and page.get('image_status'): # for reflect prompt generation
            
            if page.get("image_status") == "needs_revision":  #only rerun image prompt if image status = needs revision
                image_node_op = await _call_tool(
                    tools,
                    "generate_image_prompt",
                    page_text=page["text"],
                    existing_prompt=page["image_prompt"],
                    feedback=page["image_feedback"],
                    character_tags=char_tags,
                )
                image_prompt,_, _ , _ = unwrapped_mcpcontent(image_node_op)
                logger.info(f"[IMAGE PROMPT NODE] REVISED image prompt of page text: \n {page["text"]}\n with feedback --> \n {image_prompt}\n")
                updated.append({**page,
                                "image_prompt": image_prompt,
                               "image_feedback": "",
                                "image_status" : ""
                            })
            else:
                updated.append(page)
        
            
 
    logger.info(f"[IMAGE PROMPT NODE] Created image prompts {'for ' + str(sorted(target)) if target else '(all)'}")
    
    return {**state, "pages": updated}


# ── 7. Reflect-image (automated critique) ──────────────────────────────────── #
async def reflect_image_prompt_node(state: StorybookState, tools: dict) -> StorybookState:
    logger.info("[Reflect-Image] Critiquing image prompts...")
    print("[Reflect-Image] Critiquing image prompts...")
    
    updated: list[PageState] = []
    for page in state["pages"]:
        logger.info(f"[Reflect-Image] Page number : {page["page_number"]}")
        if page["image_revisions"] >= MAX_IMG_REVISIONS:
            updated.append({**page, "image_status" : "ok"})
            continue
        if page["image_status"] == "ok": # dont re-evalate if image status is ok
            updated.append(page)
            continue
            
        crit = await _call_tool(
            tools, "evaluate_image_prompt", page_text=page["text"], image_prompt=page["image_prompt"]
        )
        
        _,_, image_status, image_messages = unwrapped_mcpcontent(crit)
        logger.info(f'[Reflect-Image] Prompt needs revision from \n {page["image_prompt"]}\n  llm message →  \n {image_messages}\n image_status → \n {image_status}\n')
        if image_status == "needs_revision":
            updated.append(
                {
                    **page,
                    "image_feedback": image_messages,
                    "image_status" : image_status.lower(),
                    "image_revisions": page["image_revisions"] + 1,
                }
            )
        else:
            updated.append({**page, "image_status" : image_status.lower()})

    return {**state, "pages": updated}
# ── 8. Image generation using prompt ────────────────────────────────────── #
async def image_gen_node(
    state: StorybookState, tools: dict, ctx: PipelineContext, only_pages: list[int] | None = None
) -> StorybookState:
    
    ctx.stop_server()  # free GPU 1 before/while diffusion runs on GPU 0

    target = set(only_pages) if only_pages else None
    logger.info(f"[ImageGen] Generating images {'for ' + str(sorted(target)) if target else '(all)'}")
    os.makedirs("images", exist_ok=True)

    updated: list[PageState] = []
    for page in state["pages"]:
        if target and page["page_number"] not in target:
            updated.append(page)
            continue
        img_path = page.get("image_path") or f"images/{page['page_number']}.png"
        img_result = await _call_tool(tools, "generate_image", prompt=page["image_prompt"], output_path=img_path)
        img_path, _, image_status, image_message = unwrapped_mcpcontent(img_result)
        logger.info(f"[ImageGen] Page {page['page_number']}: {image_status}")
        if image_status != "ok":
            logger.warning(f"[ImageGen] Problem: {image_message} for {img_path}")
        updated.append({**page, "image_path": img_path})

    return {**state, "pages": updated}
    
# ── 9. Human review — per-page verdicts ────────────────────────────────────── #
async def human_review_pages_node(state: StorybookState) -> StorybookState:
    """
    Per-page verdicts (story text is already human-approved and is not revisited here):
        "ok"         — keep image as-is
        "image_only" — regenerate this page's image

    interrupt() payload the runner must provide:
    {
      "verdicts": {"1": "ok", "2": "image_only"},
      "notes": {"2": "needs a night scene not daytime"}
    }
    In auto-mode the runner injects all "ok".
    """
    logger.info('Human reviewing images!')
    payload: dict = interrupt(
        {
            "prompt": (
                "Provide a verdict for each page's image.\n"
                "Keys are page numbers (as strings), values are:\n"
                "'ok' → keep image as-is\n"
                "'image_only' → regenerate image for this page\n\n"
                'Format: {"verdicts": {"1":"ok","2":"image_only"}, '
                '"notes": {"2":"needs a night scene"}}'
            ),
            "pages": state["pages"],
        }
    )

    if isinstance(payload, str) and payload.strip().lower() in ("ok", ""):
        verdicts = {str(p["page_number"]): "ok" for p in state["pages"]}
        notes = {}
    else:
        verdicts = payload.get("verdicts", {})
        notes = payload.get("notes", {})

    updated: list[PageState] = []
    for page in state["pages"]:
        key = str(page["page_number"])
        verdict = verdicts.get(key, "ok")
        if verdict not in VALID_VERDICTS:
            verdict = "ok"
        note = notes.get(key, "")
        updated.append({**page, "verdict": verdict, "human_note": note})

    pages_needing_image = [p["page_number"] for p in updated if p["verdict"] == "image_only"]

    logger.info(f"→ Image reruns : {pages_needing_image or 'none'}")

    return {
        **state,
        "pages": updated,
        "pages_needing_image": pages_needing_image,
    }


# ── 10. Image rerun (only flagged pages) ─────────────────────────────────────── #
async def image_rerun_node(state: StorybookState, tools: dict, ctx: PipelineContext) -> StorybookState:
    targets = state.get("pages_needing_image", [])
    logger.info(f"[ImageRerun] Re-generating prompts + images for pages: {targets}")

    updated_pages = []
    for page in state["pages"]:
        if page["page_number"] in targets:
            updated_pages.append({**page, "image_revisions": 0, "image_feedback": "", "image_status": ""})
        else:
            updated_pages.append(page)
    state = {**state, "pages": updated_pages}

    ctx.restart_server()  # vLLM needed again for the reflect/regenerate loop

    state = await image_prompt_node(state, tools, only_pages=targets)
    state = await image_gen_node(state, tools, ctx, only_pages=targets)
    return {**state, "pages_needing_image": []}

# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════════════════════════

def should_replan(state: StorybookState) -> Literal["replan", "ok"]:   
    if (state["story_evaluation"].get("status") == "needs_revision" and state["story_revisions"] < MAX_STORY_REVISIONS):
        return "replan"
    return "ok"


def should_rewrite_story_human(state: StorybookState) -> Literal["rewrite", "ok"]:
    fb = state.get("human_story_feedback", "ok").lower().strip()
    if fb in ("ok", "yes", "approve", "approved", "looks good", ""):
        return "ok"
    state["story_rewrite_reason"] = fb
    return "rewrite"


def should_retry_image(state: StorybookState) -> Literal["retry", "ok"]:
    needs = any(
        p.get("image_feedback") and p["image_revisions"] < MAX_IMG_REVISIONS
        for p in state["pages"]
    )
    return "retry" if needs else "ok"


def route_after_human_review(state: StorybookState) -> Literal["image_only", "all_ok"]:
    return "image_only" if state.get("pages_needing_image") else "all_ok"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(ctx: PipelineContext, checkpointer=None) -> Any:
    async def _story(s):            return await story_node(s, ctx.tools)
    async def _reflect_story(s):    return await reflect_story_node(s, ctx.tools)
    async def _human_review_story(s): return await human_review_story_node(s)
    async def _extract_characters(s): return await extract_characters_node(s, ctx.tools)
    async def _split_pages(s):      return await split_pages_node(s, ctx.tools)
    async def _image_prompt(s):     return await image_prompt_node(s, ctx.tools)
    async def _reflect_image_prompt(s): return await reflect_image_prompt_node(s, ctx.tools)
    async def _image_gen(s):        return await image_gen_node(s, ctx.tools, ctx)
    async def _human_review_pages(s): return await human_review_pages_node(s)
    async def _image_rerun(s):      return await image_rerun_node(s, ctx.tools, ctx)
 

    g = StateGraph(StorybookState)

    g.add_node("story", _story)
    g.add_node("reflect_story", _reflect_story)
    g.add_node("human_review_story", _human_review_story)
    g.add_node("extract_characters", _extract_characters)
    g.add_node("split_pages", _split_pages)
    g.add_node("image_prompt", _image_prompt)
    g.add_node("reflect_image_prompt", _reflect_image_prompt)
    g.add_node("image_gen", _image_gen)
    g.add_node("human_review_pages", _human_review_pages)
    g.add_node("image_rerun", _image_rerun)

    g.set_entry_point("story")
    g.add_edge("story", "reflect_story")

    g.add_conditional_edges(
        "reflect_story", should_replan,
        {"replan": "story", "ok": "human_review_story"},
    )
    g.add_conditional_edges(
        "human_review_story", should_rewrite_story_human,
        {"rewrite": "story", "ok": "extract_characters"},
    )

    g.add_edge("extract_characters", "split_pages")
    g.add_edge("split_pages", "image_prompt")
    g.add_edge("image_prompt", "reflect_image_prompt")

    g.add_conditional_edges(
        "reflect_image_prompt", should_retry_image,
        {"retry": "image_prompt", "ok": "image_gen"},
    )

    g.add_edge("image_gen", "human_review_pages")

    g.add_conditional_edges(
        "human_review_pages", route_after_human_review,
        {"image_only": "image_rerun", "all_ok": END},
    )
    g.add_edge("image_rerun", END)

    return g.compile(checkpointer=checkpointer)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _collect_story_feedback() -> str:
    return input("\n[You] Story feedback (Enter = ok if story seems good): ").strip() or "ok"


def _collect_page_verdicts(pages: list[PageState]) -> dict:
    print("\n  Verdicts: ok | image_only")
    verdicts, notes = {}, {}
    for page in pages:
        while True:
            v = (
                input(
                    f"  Page {page['page_number']} verdict "
                    f"[{'/'.join(VALID_VERDICTS)}] (Enter=ok): "
                )
                .strip()
                .lower()
                or "ok"
            )
            if v in VALID_VERDICTS:
                verdicts[str(page["page_number"])] = v
                break
            print(f"  ✗ Invalid — choose from: {', '.join(sorted(VALID_VERDICTS))}")
        if verdicts[str(page["page_number"])] != "ok":
            n = input(f"  Page {page['page_number']} note (optional guidance): ").strip()
            if n:
                notes[str(page["page_number"])] = n
    return {"verdicts": verdicts, "notes": notes}
    


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _compose_pdf(pages, title, output_path):

    try:
        from pathlib import Path
        from fpdf import FPDF
    
        if not pages:
            return {"status": "error", "message": "No pages!"}
    
        # Font paths
        REGULAR = "/kaggle/input/datasets/anagha1992/font-notosans/static/NotoSans-Regular.ttf"
        BOLD = "/kaggle/input/datasets/anagha1992/font-notosans/static/NotoSans-Bold.ttf"
        ITALIC = "/kaggle/input/datasets/anagha1992/font-notosans/static/NotoSans-Italic.ttf"
    
        pdf = FPDF(orientation="P", unit="mm", format="A5")
        pdf.set_auto_page_break(auto=False)
    
        # Register Unicode fonts
        pdf.add_font("Noto", "", REGULAR)
        pdf.add_font("Noto", "B", BOLD)
        pdf.add_font("Noto", "I", ITALIC)

        # ---------------------------
        # Title Page
        # ---------------------------
        pdf.add_page()
        
        img = pages[0].get("image_path", "") #pages[0] has the title image
     
        if img and Path(img).exists():
            
            pdf.image(
                img,
                    x=0,
                    y=0,
                    w=pdf.w,   # page width
                    h=pdf.h    # page height
                )
    
        pdf.set_font("Noto", "B", 24)
        pdf.ln(40)
     
        pdf.multi_cell(
            w=0,          # Use available width between margins
            h=12,
            text=title,
            align="C"
        )
        
        # ---------------------------
        # Story Pages
        # ---------------------------

        import random
        pages = pages[1:] #to exclude the title page 
        for p in sorted(pages, key=lambda x: x["page_number"]):
            
            pdf.add_page()
    
            img = p.get("image_path", "")
        
    
            if img and Path(img).exists():
                
                # pdf.image(img,
                #     x=0,
                #     y=0,
                #     w=pdf.w,   # page width
                #     h=pdf.h    # page height
                # )
                pdf.image(img,
                    x=0,
                    y=0,
                    w=pdf.w,   # page width
                    h=145    # page height
                )
                ty = 108
 
            position = random.choice(["top", "middle", "bottom"])
            
            # if position == "top":
            #     ty = random.randint(15, 35)
            # elif position == "middle":
            #     ty = random.randint(65, 95)
            # else:  # bottom
            ty = random.randint(120, 145)
            
            pdf.set_xy(12, ty)
            
            text=remove_emojis(p["text"])
            max_width = 124
            font_size = 25
    
            while font_size > 18:
                pdf.set_font("Noto", "", font_size)
                if pdf.get_string_width(text) <= max_width * 2:  # about 2 lines
                    break
                font_size -= 1
            
            line_height = font_size * 0.5
            
            pdf.set_font("Noto", "", font_size)
            pdf.multi_cell(max_width, line_height, text)
            
        pdf.output(output_path)
    
        return {
            "status": "ok",
            "output": output_path,
            "pages_written": len(pages),
        }
    
    except ImportError:
        return {
            "status": "error",
            "message": "PDF Generation: fpdf2 not installed. Run: pip install fpdf2",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"PDF Generation: {e}",
        }
        
async def run_pipeline(
    theme: str,
    num_pages: int = 6,
    title: str = "",
    auto: bool = False,
 
) -> StorybookState:

    mcp_env = os.environ.copy()

    MCP_CONFIG = {
        "storybook": {
            "command": "python",
            "args": ["mcpserver.py"],
            "transport": "stdio",
            "env": mcp_env,
        }
    }
    ctx = await PipelineContext.create(MCP_CONFIG)
    
    try:
        min_sentences =  num_pages
        max_sentences =  num_pages * 3
        checkpointer = MemorySaver()
        graph = build_graph(ctx, checkpointer)
    
        mermaid = graph.get_graph().draw_mermaid()
        with open("graph.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid)
    
        config = {"configurable": {"thread_id": "run-1"}}
        initial: StorybookState = {
            "theme": theme,
            "num_pages": num_pages,
            "title": "",
            "min_sentences": min_sentences,
            "max_sentences": max_sentences,
            "story": [],
            "story_evaluation": {},
            "story_revisions": 0,
            "human_story_feedback": "",
            "story_rewrite_reason": "",
            "characters": {},
            "pages": [],
            "pages_needing_image": [],
            "pdf_path": "",
            "errors": [],
        }
    
        
        # # ── First run (stops at human_review_story) ── #
        final = await graph.ainvoke(initial, config=config)
        snap = graph.get_state(config)

        # ── Checkpoint 1: Story review loop ── #
        while snap.next and "human_review_story" in snap.next:
            ##Pull the payload the node handed to interrupt()
            interrupt_obj = snap.tasks[0].interrupts[0]
            payload = interrupt_obj.value
    
            print("\n" + "═" * 62)
            print("[Human Review — Story]")
            print(payload["prompt"])
            print('\n'.join(payload["story"]))
            print("═" * 62)
    
            fb = "ok" if auto else _collect_story_feedback()
    
            # Resume: this value becomes the return value of interrupt(...)
            final = await graph.ainvoke(Command(resume=fb), config=config)
            snap = graph.get_state(config)

        # ── Checkpoint 2: Per-page review loop ── #
        while snap.next and "human_review_pages" in snap.next:
            interrupt_obj = snap.tasks[0].interrupts[0]
            payload = interrupt_obj.value  # {"prompt": ..., "pages": [...]}
    
            if auto:
                resume_value = {
                    "verdicts": {str(p["page_number"]): "ok" for p in payload["pages"]},
                    "notes": {},
                }
                print('resume value', resume_value)
            else:
                print("\n" + "═" * 62)
                print("[Human Review — Story & Images]")
                for page in payload["pages"]:
                    print(f"\n  Page {page['page_number']}")
                    print(f"  Text   : {page['text']}")
                    print(f"  Prompt : {page['image_prompt']}")
                    img_path = page.get("image_path")
                    if img_path:
                        print(f"Image  : {img_path}")
                        display(IPImage(filename=img_path, width=350))
                    else:
                        print("Image  : (not generated)")
                print("═" * 62)
                resume_value = _collect_page_verdicts(payload["pages"])
    
            final = await graph.ainvoke(Command(resume=resume_value), config=config)
            snap = graph.get_state(config)
             
    
        logger.info(f"Final state saved :\n{final}")
        final = {'theme': "Anagha aged 26 and her niece,Rachel aged 3 having a heartful conversation about Anagha's love towards Rachel", 'num_pages': 6, 'title': "Anagha's Love Language", 'min_sentences': 6, 'max_sentences': 18, 'story': ['Anagha sat on the floor with Rachel, who was holding a stuffed bear.', "Rachel looked up and asked, 'Anagha, do you love me?'", "Anagha smiled and said, 'Of course, sweetie. I love you very much.'", "She pointed to the bear and said, 'This bear loves you too. It hugs you when you're sad.'", "Rachel hugged the bear tightly and giggled, 'I love you too, Anagha!'", "Anagha tickled Rachel's belly and said, 'You make me happy every day.'", 'They sat together on the couch, sharing a warm hug.', "Anagha whispered, 'I will always love you, my little star.'"], 'story_evaluation': {'status': 'ok', 'message': []}, 'story_revisions': 0, 'human_story_feedback': 'ok', 'story_rewrite_reason': '', 'characters': {'Anagha': {'role': 'Primary caregiver', 'visual_tags': 'Adult woman with warm brown eyes, wearing light blue sweater and jeans, holding stuffed bear, smiling warmly'}, 'Rachel': {'role': 'Child', 'visual_tags': 'Young girl with yellow dress and red bow, wearing blue sneakers, holding stuffed bear, smiling broadly'}}, 'pages': [{'page_number': 0, 'text': "Anagha aged 26 and her niece,Rachel aged 3 having a heartful conversation about Anagha's love towards Rachel", 'image_prompt': "Children's watercolor illustration. Anagha, adult with warm brown eyes, light blue sweater, jeans, holds stuffed bear, smiles warmly. Rachel, young girl in yellow dress, red bow, blue sneakers, holds stuffed bear, smiles broadly. They sit on a cozy rug, knees touching, engaged in heartfelt conversation. Scene: sunlit living room with soft textiles and wooden furniture. Lighting: warm golden hues filtering through curtains. Mood: tender and joyful. Composition: close-up, eye contact, stuffed bears on the floor. Visual details: textured fabric, gentle shadows, soft light highlighting their expressions.", 'image_path': 'images/0.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}, {'page_number': 1, 'text': "Anagha sat on the floor with Rachel, who was holding a stuffed bear. Rachel looked up and asked, 'Anagha, do you love me?'", 'image_prompt': 'Children\'s watercolor illustration. Anagha, adult woman with warm brown eyes, light blue sweater, jeans, holds stuffed bear, smiling warmly. Rachel, young girl in yellow dress, red bow, blue sneakers, holds stuffed bear, smiling broadly. Main action: Rachel looks up, asking, "Anagha, do you love me?" Scene: Cozy living room floor with soft rug and bookshelves. Lighting: Warm, soft natural light through window. Mood: Gentle, heartfelt. Composition: Close-up of their connected faces, stuffed bears between them. Important details: Textured watercolor brushstrokes, warm tones, subtle shadows around the edges.', 'image_path': 'images/1.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}, {'page_number': 2, 'text': "Anagha smiled and said, 'Of course, sweetie. I love you very much.' She pointed to the bear and said, 'This bear loves you too. It hugs you when you're sad.'", 'image_prompt': "Children's watercolor illustration. Anagha, adult woman with warm brown eyes, light blue sweater, jeans, holds stuffed bear, smiles warmly. Rachel, young girl in yellow dress, red bow, blue sneakers, holds stuffed bear, smiles broadly. Anagha points to bear, speaking to Rachel in cozy living room. Soft afternoon light filters through window. Warm, comforting mood. Composition shows close interaction, bear centered. Details: books on shelf, fluffy rug, gentle shadows.", 'image_path': 'images/2.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}, {'page_number': 3, 'text': "Rachel hugged the bear tightly and giggled, 'I love you too, Anagha!'", 'image_prompt': "Children's watercolor illustration. Anagha, adult woman with warm brown eyes, light blue sweater, jeans, holds stuffed bear, smiles warmly. Rachel, young girl in yellow dress, red bow, blue sneakers, holds stuffed bear, smiles broadly. Rachel hugs bear tightly, giggles, embraces Anagha. Scene: cozy living room with soft blankets, wooden shelves. Warm golden lighting. Mood: affectionate, joyful. Composition: side profile of both characters, window light illuminating faces. Important details: bears’ embroidered patches, Rachel’s bow, Anagha’s sweater texture.", 'image_path': 'images/3.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}, {'page_number': 4, 'text': "Anagha tickled Rachel's belly and said, 'You make me happy every day.'", 'image_prompt': 'Children’s watercolor illustration. Anagha, adult woman in light blue sweater and jeans, holds stuffed bear while tickling Rachel’s belly. Rachel, young girl in yellow dress and red bow, giggles with blue sneakers. Cozy living room scene with soft sunlight streaming through window. Warm, golden lighting bathes the room. Joyful, heartfelt mood. Close-up composition focusing on their interaction. Highlight warm tones, stuffed bear, and gentle shadows from window light.', 'image_path': 'images/4.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}, {'page_number': 5, 'text': 'They sat together on the couch, sharing a warm hug.', 'image_prompt': "Children's watercolor illustration. Anagha (adult, warm brown eyes, light blue sweater, jeans, holding stuffed bear, smiling warmly) and Rachel (young girl, yellow dress, red bow, blue sneakers, holding stuffed bear, smiling broadly) sit on a cozy couch, hugging warmly. Scene: sunlit living room with soft blankets and plush cushions. Lighting: warm golden tones. Mood: comforting. Composition: close-up of their embrace, emphasizing connection. Details: textured paper, gentle shadows, vibrant colors.", 'image_path': 'images/5.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}, {'page_number': 6, 'text': "Anagha whispered, 'I will always love you, my little star.'", 'image_prompt': 'Children’s watercolor illustration. Anagha, adult woman with warm brown eyes, light blue sweater, jeans, holds stuffed bear, smiles warmly. Rachel, young girl in yellow dress, red bow, blue sneakers, holds stuffed bear, smiles broadly. Anagha whispers lovingly to Rachel in a cozy living room. Soft golden light spills through a window, casting gentle shadows. Warm, tender mood. Composition features both characters at eye level, with Rachel gazing up. Important details: textured fabric, soft light, stuffed bears, warm color palette.', 'image_path': 'images/6.png', 'image_feedback': '', 'image_revisions': 0, 'verdict': 'ok', 'human_note': '', 'image_status': 'ok'}], 'pages_needing_image': [], 'pdf_path': '', 'errors': []}
        pdf_path = final.get('pdf_path') or f"{final.get('title')}.pdf"
        pdf_results = _compose_pdf(final.get('pages', []), final.get('title'), pdf_path)
        if pdf_results.get("status", "") != "ok":
            final.setdefault("errors", []).append(pdf_results.get("message", ""))
        final["pdf_path"] = pdf_path 
        
        print("\n" + "═" * 62)
        print("STORYBOOK COMPLETE")
        print(f"  Title      : {final.get('title')}")
        print(f"  Characters : {list(final.get('characters', {}).keys())}")
        print(f"  Pages      : {len(final.get('pages', []))}")
        print(f"  PDF        : {final.get('pdf_path') or '(not generated)'}")
        if final.get("errors"):
            print(f"  Errors     : {final['errors']}")
        print("═" * 62)
    finally:
        ctx.shutdown()
    
       

if __name__ == "__main__":
    
    try:
        #--------------INPUTS-------------
        theme = "Anagha aged 26 and her niece,Rachel aged 3 having a heartful conversation about Anagha's love towards Rachel"
        num_pages = 6
        auto = True
        #---------------------------------
        
        start_time = time.time() 
        await run_pipeline(
            theme= theme,
            num_pages=num_pages,
            auto= auto
            )
        
        print(f"Agent took approximately {time.time()- start_time} second")
    
    except ExceptionGroup as eg:
        for sub in eg.exceptions:
            print(repr(sub))